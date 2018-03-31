# PID Controller

## The effect of P (Proportion)

P is the simplest form of controller. It steers in negative preprotion to the cross track error. This brings the car towards the path. This effect is instantaneous and only reacts to the current error.

When P is very high it turns hard towards the planned path. This caused it to get to the path quickly on stright sections but due to the large turning angles it caused drastic overshoot which caused massive ossilations around the path, eventually going off the road.

When P is very low the car swerves across the road as it does try to turn toward the path. When the other terms are tuned well the car still turn toward the path slowly but the massive swings would not be acceptable to ride in.

The model uses a value of 0.2 for P.

## The effect of I (Integral)

I is based on the integral of the error over time. This means that it can take account for errors over time. This finds systematic errors such as the car having a defect and pulling to the right.

When I is very high the sum of normal errors start to cause a changes in the steering error. This means that over time the error gets larger and larger and eventually the car starts steering more erradically.

When I is very low and a systematic error is added (constant steering angle is always added) then the model will very get to the path and instead converges to the path plus and error.

The model uses a value of 0.004 for I.

I also added a systematic error to the steering angle so that the I term is more useful and to show off that it works.

## The effect of D (Derivative)

D is the effect of the change in error. The derivative is defined as the change of error over time. This means that as the change in the error (the car is apporching the path) gets smaller then the car will counter steer, steer away from the path. This is will means that the car shouldn't over shoot as much.

When D is very high the car counter steers too much and ends up turning away from the path.

When D is very low the car doesn't counter steer at all and ends up oscillating around the path which would cause see sickness.

The model uses a value of 3.0 for D.

## Parameter selection.

The parameters if the model were choosen with using manual tuning.
