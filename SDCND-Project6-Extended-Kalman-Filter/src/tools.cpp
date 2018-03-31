#include <iostream>
#include "tools.h"
#include <math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

#define EPSLION 0.00001

VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    /**
     * Calculate the RMSE here.
     */
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
        cout << "Invalid estimations or ground_truth data" << endl;
        return rmse;
    }

    for (int i = 0; i < estimations.size(); i++) {
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array() * residual.array();
        rmse += residual;
    }

    rmse = rmse / estimations.size();

    rmse = rmse.array().sqrt();

    return rmse;
}

MatrixXd CalculateJacobian(const VectorXd& x_state) {
    /**
     * Calculate a Jacobian here.
     */
    MatrixXd Hj(3, 4);

    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    double px2 = px * px;
    double py2 = py * py;

    if (fabs(px2) < EPSLION) {
      px2 = EPSLION;
    }
    if (fabs(py2) < EPSLION) {
      py2 = EPSLION;
    }

    double ss = px2 + py2;
    double srss = sqrt(ss);
    double cached_dem = ss * srss;
    double pyvx = py * vx;
    double pxvy = px * vy;

    Hj(0, 0) = px / srss;
    Hj(0, 1) = py / srss;
    Hj(0, 2) = 0;
    Hj(0, 3) = 0;

    Hj(1, 0) = -py / ss;
    Hj(1, 1) = px / ss;
    Hj(1, 2) = 0;
    Hj(1, 3) = 0;

    Hj(2, 0) = (py * (pyvx - pxvy)) / (cached_dem);
    Hj(2, 1) = (px * (pxvy - pyvx)) / (cached_dem);
    Hj(2, 2) = px / srss;
    Hj(2, 3) = py / srss;

    return Hj;
}

void PolarToCartesian(const MeasurementPackage &measurement_pack, double &px, double &py) {
    double rho = measurement_pack.raw_measurements_[0];
    double phi = measurement_pack.raw_measurements_[1];

    px = rho * cos(phi);
    py = rho * sin(phi);
}

Eigen::VectorXd CartesianToPolar(const Eigen::VectorXd x) {
    VectorXd z_pred(3);

    double px = x(0);
    double py = x(1);
    double vx = x(2);
    double vy = x(3);

    if (fabs(px) < EPSLION) {
      px = EPSLION;
    }

    double px2 = px * px;
    double py2 = py * py;
    double rho = sqrt(px2 + py2);

    if (fabs(rho) < EPSLION) {
      rho = EPSLION;
    }

    double phi = atan2(py, px);

    z_pred[0] = rho;
    z_pred[1] = phi;
    z_pred[2] = (px * vx + py * vy) / rho;

    return z_pred;
}

void NormalizeAngle(double &angle) {
    while (angle < -M_PI or angle > M_PI) {
        if (angle < -M_PI) {
            angle += 2 * M_PI;
        } else {
            angle -= 2 * M_PI;
        }
    }
}
