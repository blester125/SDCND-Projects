# Global Kinematic Model
##

### The Model

This model is the Global Kinematic Model as presented in the Lectures.

The model includes the vehicle's state, (x and y position, orientation (psi), and velocity) and the model error (cross track error, and orientation error (epsi)). The model includes actuators that control the steering angle (delta) and the acceleration. The model calculates the values for the next time step with the following equations.

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20x_%7Bt&plus;1%7D%20%26%3D%20x_t%20&plus;%20v_t%20*%20%5Ccos%28%5Cpsi%29%20*%20dt%20%5C%5C%20y_%7Bt&plus;1%7D%20%26%3D%20y_t%20&plus;%20v_t%20*%20%5Csin%28%5Cpsi%29%20*%20dt%20%5C%5C%20%5Cpsi_%7Bt&plus;1%7D%20%26%3D%20%5Cpsi_t%20+%20%5Cfrac%7Bv_t%7D%7BL_f%7D%20*%20%5Cdelta%20*%20dt%20%5C%5C%20v_%7Bt&plus;1%7D%20%26%3D%20v_t%20&plus;%20a%20*%20dt%20%5C%5C%20cte_t%20%26%3D%20f%28x%29%20-%20y_t%20%5C%5C%20cte_%7Bt&plus;1%7D%20%26%3D%20cte_t%20&plus;%20v_t%20*%20sin%28e%5Cpsi%29%20*%20dt%20%5C%5C%20cte_%7Bt&plus;1%7D%20%26%3D%20f%28x%29%20-%20y_t%20&plus;%20%28v_t%20*%20sin%28e%5Cpsi%29%20*%20dt%29%20%5C%5C%20%5Cpsi%20des_t%20%26%3D%20arctan%28f%27%28x%29%29%20%5C%5C%20e%5Cpsi_t%20%26%3D%20%5Cpsi_t%20-%20%5Cpsi%20des_t%20%5C%5C%20e%5Cpsi_%7Bt&plus;1%7D%20%26%3D%20e%5Cpsi_t%20&plus;%20%28%5Cfrac%7Bv_t%7D%7BL_f%7D%20*%20%5Cdelta%20*%20dt%29%5C%5C%20%5Cend%7Balign*%7D)

### Timestep Length and Elapsed Duration.

The value for N is 10 and for dt is 0.1. These were chosen based on office hours for the project. This results in the optimizer considering a 1 second duration. Other values were things like 8 and 0.124 and 6 and 0.15. None of these values worked much better.

### Polynomial Fitting

I transformed the global way points to be relative to the vehicle. This made the x, y, and psi values for the car be 0, 0 and 0.

### MPC with Latency

The model values are updated with each time step. The delay of the actuators is 100ms. This matches the timestep interval I use meaning that each actuation is simply applied one timestep later. This is simple to account for by tweaking the equations.
