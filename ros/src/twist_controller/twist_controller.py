from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
     accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # Init yaw controller
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        # PID Params
        kp = 0.3
        ki = 0.1
        kd = 0.0
        mn = 0.0 # Min throttle value
        mx = 0.2 # Maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        # LP filter params
        tau = 0.5
        ts = 0.02
        self.vel_lpf = LowPassFilter(tau, ts)

        # Other infos
        self.vehice_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit # Unused
        self.accel_limit = accel_limit # Unused
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()


    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):

        # If manual control is enabled, don't do anything
        if not dbw_enabled:
            # Avoid error acuumulation
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0

        # Filter current velocity
        current_vel = self.vel_lpf.filt(current_vel)

        # Calculate steering angle
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        # Check what is current vel error from target vel
        vel_error = linear_vel - current_vel
        # Update last vel
        self.last_vel = current_vel

        # Calculate time step (needed for PID)
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        # Use PID to calculate throttle
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        # If target velocity is 0 and current velocity is very small that means we need to stop
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            # Apply brake
            brake = 400 # N * m
        # If throttle is smal and we are going over target velocity
        elif throttle < 0.1 and vel_error < 0:
            # Let go of throttle
            throttle = 0
            decel = max(vel_error, self.decel_limit)

            # apply brake - slightly
            brake = abs(decel) * self.vehice_mass * self.wheel_radius # Torque N * m

        return throttle, brake, steering