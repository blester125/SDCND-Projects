#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"
#include "measurement_package.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

/**
 * A helper method to calculate RMSE.
 */
VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

/**
 * A helper method to calculate Jacobians.
 */
MatrixXd CalculateJacobian(const VectorXd& x_state);

/**
 * A Helper method that converts Polar to Cartesian coordinates
 */
void PolarToCartesian(const MeasurementPackage &measurement_pack, double &px, double &py);

/**
 * A Helper method to convert Cartesian To Polar coordinates
 */
Eigen::VectorXd CartesianToPolar(const Eigen::VectorXd x);

/**
 * A Helper method to normalize an angle to be between -pi and pi
 */
void NormalizeAngle(double &angle);

#endif /* TOOLS_H_ */
