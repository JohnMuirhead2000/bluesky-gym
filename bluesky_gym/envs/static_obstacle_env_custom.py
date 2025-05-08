import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

## NOTEWORTH ODDIDES

# degrees seems to be counting up counter-clockwise
# the display has X going up donw and Y going left-right

import gymnasium as gym
from gymnasium import spaces

DISTANCE_MARGIN = 5 # km (make it larger becuase we are hitting a 3D target now)
ALT_MEAN = 1500
ALT_STD = 3000
VZ_MEAN = 0
VZ_STD = 5

REACH_REWARD = 1 # reach set waypoint
DRIFT_PENALTY = -0.01
RESTRICTED_AREA_INTRUSION_PENALTY = -5

INTRUSION_DISTANCE = 5 # NM

WAYPOINT_DISTANCE_MIN = 100 # KM
WAYPOINT_DISTANCE_MAX = 170 # KM

WAYPOINT_ALT_MAX = 11000 # in meters
ACTION_2_MS = 12.5  # approx 2500 ft/min

OBSTACLE_DISTANCE_MIN = 20 # KM
OBSTACLE_DISTANCE_MAX = 150 # KM

D_HEADING = 45 #degrees
D_SPEED = 20/3 # kts (check)
V_SPEED = 20/3 # kts (check) TODO confirm this value is correct. 

AC_SPD = 150 # kts
MAX_ALTITUDE = 11000  # in meters

NM2KM = 1.852
NM2F = 60.76
FL2F = 100
MpS2Kt = 1.94384
VMpS2Kt = 1.94384 # TODO investigate what this is
M2FL = .0328 #M to flight level
M2KM = .001
NM2M = 1852
F2M = .3048

ACTION_FREQUENCY = 10

NUM_OBSTACLES = 10
NUM_WAYPOINTS = 1

OBSTACLE_AREA_RANGE = (50, 1000) # In NM^2
CENTER = (51.990426702297746, 4.376124857109851) # TU Delft AE Faculty coordinates

MAX_DISTANCE = 351 # width of screen in km

class StaticObstacleEnv(gym.Env):
    """ 
    Static Obstacle Conflict Resolution Environment

    TODO:
    - Investigate CNN and Lidar based observation
    - Change rendering such that none-square screens are also possible
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None, test_mode=False, test_dictionary={}):

        self.test_mode = test_mode
        self.test_dictionary = test_dictionary

        self.window_width = 512 # pixels
        self.window_height = 512 # pixels
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        if self.test_mode and self.test_dictionary["building_mode"][0] == "no_building":
            self.num_of_obstacles = 0
        elif self.test_mode and self.test_dictionary["building_mode"][0] == "square":
            self.num_of_obstacles = 1
        else:
            self.num_of_obstacles = NUM_OBSTACLES

        self.observation_space = spaces.Dict(
            {
                "altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64), # ownship vertical speed
                "altitude_differences": spaces.Box(-np.inf, np.inf, shape = (self.num_of_obstacles,), dtype=np.float64), # 
                "vz": spaces.Box(-np.inf, np.inf, dtype=np.float64), # ownship vertical speed  
                "altitude_waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64), 
                "destination_waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "destination_waypoint_cos_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "destination_waypoint_sin_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "restricted_area_radius": spaces.Box(0, 1, shape = (self.num_of_obstacles,), dtype=np.float64),
                "restricted_area_distance": spaces.Box(-np.inf, np.inf, shape = (self.num_of_obstacles, ), dtype=np.float64),
                "cos_difference_restricted_area_pos": spaces.Box(-np.inf, np.inf, shape = (self.num_of_obstacles,), dtype=np.float64),
                "sin_difference_restricted_area_pos": spaces.Box(-np.inf, np.inf, shape = (self.num_of_obstacles,), dtype=np.float64),
            }
        )
       
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        bs.init(mode='sim', detached=True)

        # initialize dummy screen and set correct sim speed
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF')
        
        # variables for logging
        self.total_reward = 0
        self.waypoint_reached = 0
        self.crashed = 0
        self.average_drift = np.array([])

        self.obstacle_names = []

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        bs.traf.reset()

        # reset logging variables 
        self.total_reward = 0
        self.waypoint_reached = 0
        self.crashed = 0
        self.average_drift = np.array([])

        min_z = MAX_ALTITUDE*.25 # This value has to be in meters. 
        max_z = MAX_ALTITUDE*.75 # This value has to be in meters. 
        bs.traf.cre('KL001', actype="A320", acspd=AC_SPD, acalt=np.random.uniform(min_z, max_z))
        bs.stack.stack("VNAV KL001 OFF")

        # defining screen coordinates
        # defining the reference point as the top left corner of the SQUARE screen
        # from the initial position of the aircraft which is set to be the centre of the screen
        ac_idx = bs.traf.id2idx('KL001')
        d = np.sqrt(2*(MAX_DISTANCE/2)**2) #KM
        lat_ref_point,lon_ref_point = bs.tools.geo.kwikpos(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 315, d/NM2KM)
        
        self.screen_coords = [lat_ref_point,lon_ref_point]#[52.9, 2.6]

        self._generate_obstacles()
        self._generate_waypoint()

        ac_idx = bs.traf.id2idx('KL001')
        self.initial_wpt_qdr, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat[0], self.wpt_lon[0])
        bs.traf.hdg[ac_idx] = self.initial_wpt_qdr
        bs.traf.ap.trk[ac_idx] = self.initial_wpt_qdr
        observation = self._get_obs()

        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self._get_action(action)

        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
            reward, done, terminated = self._get_reward()
            if self.render_mode == "human":
                self._render_frame()
            if terminated or done:
                observation = self._get_obs()
                self.total_reward += reward
                info = self._get_info()
                return observation, reward, done, terminated, info

        observation = self._get_obs()
        self.total_reward += reward
        info = self._get_info()

        return observation, reward, done, terminated, info

    def _generate_polygon(self, centre):
        poly_area = np.random.randint(OBSTACLE_AREA_RANGE[0]*2, OBSTACLE_AREA_RANGE[1])
        R = np.sqrt(poly_area/ np.pi)
        p = [fn.random_point_on_circle(R) for _ in range(3)] # 3 random points to start building the polygon
        p = fn.sort_points_clockwise(p)
        p_area = fn.polygon_area(p)
        
        while p_area < OBSTACLE_AREA_RANGE[0]:
            p.append(fn.random_point_on_circle(R))
            p = fn.sort_points_clockwise(p)
            p_area = fn.polygon_area(p)
        
        p = [fn.nm_to_latlong(centre, point) for point in p] # Convert to lat/long coordinateS
        return p_area, p, R
    
    # this function builds obstacle_names, obstacle_vertices and obstacle_radius for us. Add obstacle_height
    def _generate_obstacles(self): 
        # Delete existing obstacles from previous episode in BlueSky
        for name in self.obstacle_names:
            bs.tools.areafilter.deleteArea(name)

        self.obstacle_names = []
        self.obstacle_vertices = []
        self.obstacle_radius = []
        self.obstacle_height = []

        # Generate coordinates for center of obstacles
        self._generate_coordinates_centre_obstacles()

        if self.test_mode and self.test_dictionary.get("building_mode")[0] == "no_building":
            return

        if self.test_mode and self.test_dictionary.get("building_mode")[0] == "square":
            # Define square size in nautical miles (NM)
            side_length = 10  # fixed side length in NM (example size)
            half_side = side_length / 2

            # Define square centered at (0, 0) in local coordinates (lat/lon)
            square_nm = [
                [-half_side, -half_side],  # Bottom-left
                [ half_side, -half_side],  # Bottom-right
                [ half_side,  half_side],  # Top-right
                [-half_side,  half_side]   # Top-left
            ]
            
            # Store the square coordinates (in NM)
            self.poly_points = np.array(square_nm)
            
            # Calculate the area (in NM²) for the square (optional)
            self.poly_area = fn.polygon_area(square_nm)

            # Convert the square's local coordinates to lat/lon coordinates
            square_latlon = [fn.nm_to_latlong(CENTER, point) for point in square_nm]
            points = [coord for point in square_latlon for coord in point]  # Flatten the list

            # Define the area in BlueSky (use the same name for consistency)
            bs.tools.areafilter.defineArea('square_building_1', 'POLY', points)

            # Store additional information (optional)
            self.obstacle_names.append('square_building_1')
            self.obstacle_vertices.append(square_latlon)
            self.obstacle_radius.append(side_length)  # Side length can be used as the radius (or modify as needed)
            self.obstacle_height.append(self.test_dictionary["building_mode"][1])  # Altitude from building mode. Assume its in FL

            return


        else:
            # Generate random obstacles if building mode is not square
            for i in range(self.num_of_obstacles):
                centre_obst = (self.obstacle_centre_lat[i], self.obstacle_centre_lon[i])
                _, p, R = self._generate_polygon(centre_obst)
                
                points = [coord for point in p for coord in point]  # Flatten the list of points
                poly_name = 'restricted_area_' + str(i + 1)
                bs.tools.areafilter.defineArea(poly_name, 'POLY', points)
                self.obstacle_names.append(poly_name)

                # Process the obstacle vertices and add them to the list
                obstacle_vertices_coordinates = []
                for k in range(0, len(points), 2):
                    obstacle_vertices_coordinates.append([points[k], points[k + 1]])

                self.obstacle_vertices.append(obstacle_vertices_coordinates)
                self.obstacle_radius.append(R)
                self.obstacle_height.append(np.random.uniform(0, MAX_ALTITUDE*M2FL))  # Random height for each obstacle


    def _generate_waypoint(self, acid = 'KL001'):
        # original _generate_waypoints function from horizotal_cr_env
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_alt = []
        self.wpt_reach = []

        ac_idx = bs.traf.id2idx(acid)
        check_inside_var = True
        loop_counter = 0
        while check_inside_var:
            loop_counter += 1
            wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
            wpt_hdg_init = np.random.randint(0, 360)
            wpt_lat, wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)
            wpt_alt = np.random.rand() * WAYPOINT_ALT_MAX*M2FL*.1 # arbitarly put the waypoint in the bottom 5th of the altitude range. 

            inside_temp = []

            #TODO double check this
            for j in range(self.num_of_obstacles):
                inside_temp.append(bs.tools.areafilter.checkInside(self.obstacle_names[j], np.array([wpt_lat]), np.array([wpt_lon]), np.array([bs.traf.alt[ac_idx]]))[0])
            check_inside_var = any(x == True for x in inside_temp)
      
            if loop_counter > 1000:
                raise Exception("No waypoints can be generated outside the obstacles. Check the parameters of the obstacles in the definition of the scenario.")

        self.wpt_lat.append(wpt_lat)
        self.wpt_lon.append(wpt_lon)
        self.wpt_alt.append(wpt_alt)
        self.wpt_reach.append(0)

    def _generate_coordinates_centre_obstacles(self, acid = 'KL001'):
        self.obstacle_centre_lat = []
        self.obstacle_centre_lon = []
        
        for i in range(self.num_of_obstacles):
            obstacle_dis_from_reference = np.random.randint(OBSTACLE_DISTANCE_MIN, OBSTACLE_DISTANCE_MAX)
            obstacle_hdg_from_reference = np.random.randint(0, 360)
            ac_idx = bs.traf.id2idx(acid)

            obstacle_centre_lat, obstacle_centre_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], obstacle_dis_from_reference, obstacle_hdg_from_reference)    
            self.obstacle_centre_lat.append(obstacle_centre_lat)
            self.obstacle_centre_lon.append(obstacle_centre_lon)

    def _get_obs(self):
        ac_idx = bs.traf.id2idx('KL001')

        self.destination_waypoint_distance = []
        self.wpt_qdr = []
        self.destination_waypoint_cos_drift = []
        self.destination_waypoint_sin_drift = []
        self.destination_waypoint_drift = []

        self.obstacle_centre_distance = []
        self.obstacle_centre_cos_bearing = []
        self.obstacle_centre_sin_bearing = []

            
        self.ac_hdg = bs.traf.hdg[ac_idx]
        self.ac_tas = bs.traf.tas[ac_idx]

        self.altitude = bs.traf.alt[ac_idx]*M2FL
        self.vz = bs.traf.vs[ac_idx]
        self.altitude_differences = []

        wpt_qdr, flat_wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat[0], self.wpt_lon[0])
        self.wpt_alt_dif = abs(self.wpt_alt[0] - self.altitude)*(1/M2FL)
        wpt_dis = np.sqrt((flat_wpt_dis*(NM2M))**2 + self.wpt_alt_dif**2)
        
        wpt_alt_dif = np.array([self.wpt_alt_dif]) / MAX_ALTITUDE

        self.destination_waypoint_distance.append(wpt_dis * M2KM)
        self.wpt_qdr.append(wpt_qdr)


        drift = self.ac_hdg - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)

        self.destination_waypoint_drift.append(drift)
        self.destination_waypoint_cos_drift.append(np.cos(np.deg2rad(drift)))
        self.destination_waypoint_sin_drift.append(np.sin(np.deg2rad(drift)))

        obs_altitude = np.array([self.altitude/(MAX_ALTITUDE*M2FL)]) # this will always put obs_sltitude between 0 and 1.
        obs_vz = np.array([(self.vz) / 15])
        
        for obs_idx in range(self.num_of_obstacles):
            obs_centre_qdr, obs_centre_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.obstacle_centre_lat[obs_idx], self.obstacle_centre_lon[obs_idx])
            obs_centre_dis = obs_centre_dis * NM2KM #KM        

            bearing = self.ac_hdg - obs_centre_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.obstacle_centre_distance.append(obs_centre_dis)
            self.obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))

            self.altitude_differences.append(self.obstacle_height[obs_idx] - self.altitude) # P sure both in in FL
            
            
        # obs_altitude = np.array([self.altitude/(MAX_ALTITUDE*M2FL)]) # this will always put obs_sltitude between 0 and 1.

        observation = {
                "altitude": obs_altitude,
                "altitude_differences": np.array(self.altitude_differences)/MAX_ALTITUDE, # 
                "vz": obs_vz,
                "altitude_waypoint_distance": wpt_alt_dif, # issue here
                "destination_waypoint_distance": np.array(self.destination_waypoint_distance)/WAYPOINT_DISTANCE_MAX,
                "destination_waypoint_cos_drift": np.array(self.destination_waypoint_cos_drift),
                "destination_waypoint_sin_drift": np.array(self.destination_waypoint_sin_drift),
                "restricted_area_radius": np.array(self.obstacle_radius)/(OBSTACLE_AREA_RANGE[0]),
                "restricted_area_distance": np.array(self.obstacle_centre_distance)/WAYPOINT_DISTANCE_MAX,
                "cos_difference_restricted_area_pos": np.array(self.obstacle_centre_cos_bearing),
                "sin_difference_restricted_area_pos": np.array(self.obstacle_centre_sin_bearing),
            }
        return observation
    
    def _get_info(self):
        return {
            'total_reward': self.total_reward,
            'waypoint_reached': self.waypoint_reached,
            'crashed': self.crashed,
            'average_drift': self.average_drift.mean()
        }

    def _get_reward(self):
        reach_reward = self._check_waypoint()
        drift_reward = self._check_drift()
        intrusion_reward, intrusion_terminate = self._check_intrusion()
        
        total_reward = reach_reward + drift_reward + intrusion_reward
        
        done = 0
        if self.wpt_reach[0] == 1:
            done = 1
        elif intrusion_terminate:
            done = 1

        # Always return truncated as False, as timelimit is managed outside
        return total_reward, done, False
    
    def _check_waypoint(self):
        reward = 0
        index = 0
        for distance in self.destination_waypoint_distance:
            if distance < DISTANCE_MARGIN and self.wpt_reach[index] != 1:
                self.waypoint_reached = 1
                self.wpt_reach[index] = 1
                reward += REACH_REWARD
                index += 1
            else:
                reward += 0
                index += 1
        return reward

    def _check_drift(self):
        drift = abs(np.deg2rad(self.destination_waypoint_drift[0]))
        self.average_drift = np.append(self.average_drift, drift)
        return drift * DRIFT_PENALTY


    # TODO double check this function!
    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        terminate = 0
        for obs_idx in range(self.num_of_obstacles):
            if bs.tools.areafilter.checkInside(self.obstacle_names[obs_idx], np.array([bs.traf.lat[ac_idx]]), np.array([bs.traf.lon[ac_idx]]), np.array([bs.traf.alt[ac_idx]])):
                if self.obstacle_height[obs_idx] > self.altitude: # self.obstacle_height and self.altitude are in FL
                    reward += RESTRICTED_AREA_INTRUSION_PENALTY
                    self.crashed = 1
                    terminate = 1
        return reward, terminate

    def _get_action(self,action):
        dh = action[0] * D_HEADING
        dv = action[1] * D_SPEED
        vs = action[2] * ACTION_2_MS
        heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx('KL001')] + dh)
        speed_new = (bs.traf.cas[bs.traf.id2idx('KL001')] + dv) * MpS2Kt

        bs.stack.stack(f"HDG {'KL001'} {heading_new}")
        bs.stack.stack(f"SPD {'KL001'} {speed_new}")

        # The actions are then executed through stack commands;
        if vs >= 0:
            bs.traf.selalt[0] = 10000000 # High target altitude to start climb
            bs.traf.selvs[0] = vs
        elif vs < 0:
            bs.traf.selalt[0] = 0 # Low target
            bs.traf.selvs[0] = vs

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        screen_coords = self.screen_coords

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))

        px_per_km = self.window_width/MAX_DISTANCE

        # draw ownship
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        heading_end_x = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/MAX_DISTANCE)*self.window_width
        heading_end_y = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/MAX_DISTANCE)*self.window_width

        qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        x_actor = ((np.sin(np.deg2rad(qdr))*dis*NM2KM) / MAX_DISTANCE)*self.window_width
        y_actor = ((-np.cos(np.deg2rad(qdr))*dis*NM2KM) / MAX_DISTANCE)*self.window_width
        pygame.draw.line(canvas,
            (235, 52, 52),
            (x_actor, y_actor),
            (x_actor+heading_end_x, y_actor-heading_end_y),
            width = 5
        )

        # Draw coordinatres:

        # Get agent position data
        lat = bs.traf.lat[ac_idx]
        lon = bs.traf.lon[ac_idx]
        alt = bs.traf.alt[ac_idx]*M2FL

        # Format text
        font = pygame.font.Font(None, 24)  # Default font, size 24
        lat_text = font.render(f"Lat: {lat:.4f}", True, (0, 0, 0))
        lon_text = font.render(f"Lon: {lon:.4f}", True, (0, 0, 0))
        alt_text = font.render(f"Alt: {alt:.0f} FL", True, (0, 0, 0))

        # Define box position and size
        padding = 10
        box_width = 150
        box_height = 60
        box_x = self.window_width - box_width - padding
        box_y = padding

        # Draw background box
        pygame.draw.rect(canvas, (255, 255, 255), (box_x, box_y, box_width, box_height))
        pygame.draw.rect(canvas, (0, 0, 0), (box_x, box_y, box_width, box_height), 1)  # Border

        # Blit text onto canvas
        canvas.blit(lat_text, (box_x + 5, box_y + 5))
        canvas.blit(lon_text, (box_x + 5, box_y + 25))
        canvas.blit(alt_text, (box_x + 5, box_y + 45))

        # draw heading line
        heading_length = 50
        heading_end_x = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/MAX_DISTANCE)*self.window_width
        heading_end_y = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/MAX_DISTANCE)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (x_actor,y_actor),
            (x_actor+heading_end_x, y_actor-heading_end_y),
            width = 1
        )

        # draw obstacles
        for idx, vertices in enumerate(self.obstacle_vertices):
            points = []
            for coord in vertices:
                lat_ref = coord[0]
                lon_ref = coord[1]
                qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat_ref, lon_ref)
                dis = dis * NM2KM
                x_ref = (np.sin(np.deg2rad(qdr)) * dis) / MAX_DISTANCE * self.window_width
                y_ref = (-np.cos(np.deg2rad(qdr)) * dis) / MAX_DISTANCE * self.window_width
                points.append((x_ref, y_ref))

            # Draw the obstacle polygon
            pygame.draw.polygon(canvas, (0, 0, 0), points)

            # Draw height in the center of the polygon
            if points:
                x_coords, y_coords = zip(*points)
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)

                height = self.obstacle_height[idx]  # get height for this obstacle
                font = pygame.font.Font(None, 16)
                height_text = font.render(f"{height:.0f} FL", True, (255, 255, 255))
                canvas.blit(height_text, (center_x - 10, center_y - 8))

        indx = 0
        for lat, lon, reach in zip(self.wpt_lat, self.wpt_lon, self.wpt_reach):
            
            alt = self.wpt_alt[indx]
            indx += 1
            qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat, lon)

            circle_x = ((np.sin(np.deg2rad(qdr)) * dis * NM2KM) / MAX_DISTANCE) * self.window_width
            circle_y = (-(np.cos(np.deg2rad(qdr)) * dis * NM2KM) / MAX_DISTANCE) * self.window_width

            color = (255, 255, 255)

            # Draw waypoint circle
            pygame.draw.circle(canvas, color, (circle_x, circle_y), radius=4, width=0)

            # Draw distance margin circle
            pygame.draw.circle(canvas, color, (circle_x, circle_y), radius=(DISTANCE_MARGIN / MAX_DISTANCE) * self.window_width, width=2)

            # --- Draw Altitude Label ---
            alt_text = f"{int(alt)} FL"  # or use another format like f"{alt:.0f} ft"
            text_surface = font.render(alt_text, True, color)
            
            # Offset the text a bit to the right and above the circle
            text_offset_x = 10
            text_offset_y = -10
            canvas.blit(text_surface, (circle_x + text_offset_x, circle_y + text_offset_y))


        self.window.blit(canvas, canvas.get_rect())
        
        pygame.display.update()
        
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pass

    # returns [lat, lon,] of the ownship
    def get_ownship_position(self):
        ac_idx = bs.traf.id2idx('KL001')
        longitude = bs.traf.lon[ac_idx]
        latitude = bs.traf.lat[ac_idx]
        altitude = bs.traf.alt[ac_idx]*M2FL #bs.traf.alt comes back in meters
        return (longitude, latitude, altitude)
    
    # returns [lat, lon,] of the ownship
    def get_ownship_heading(self):
        ac_idx = bs.traf.id2idx('KL001')
        heading = bs.traf.hdg[ac_idx]
        return heading
    
    def move_ownship(self, info): # this command takes in meters for altitude
        # info = [lat,lon,alt,hdg,spd,vspd]
        x = info[0]
        y = info[1]
        z = info[2]
        heading = info[3]
        spd = info[4]
        vspd = info[5]
        bs.stack.stack(f"MOVE {'KL001'} {x} {y} {z} {heading} {spd} {vspd}")
        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
        if self.render_mode == "human":           
            self._render_frame()

    def move_waypoint(self, info):
        # info = [lat,lon,alt]
        x = info[0]
        y = info[1]
        z = info[2] # give it in feet. We need it in NM
        lat, lon = fn.nm_to_latlong(CENTER, [x,y])
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_alt = []
        self.wpt_reach = []

        self.wpt_lat.append(lat)
        self.wpt_lon.append(lon)
        self.wpt_alt.append(z)
        self.wpt_reach.append(0)
        if self.render_mode == "human":              
            self._render_frame()

    def get_waypoint_pos(self):
        lat = self.wpt_lat[-1]
        lon = self.wpt_lon[-1]
        alt = self.wpt_lon[-1]
        # x, y = fn.latlong_to_nm(CENTER, [lat, lon])
        return [lon, lat, alt]
        