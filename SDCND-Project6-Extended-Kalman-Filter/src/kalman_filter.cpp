#include "kalman_filter.h"
#include <iostream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::SetF(const float dt) {
    F_ = MatrixXd(4, 4);
    F_ << 1, 0, dt, 0,
          0, 1, 0, dt,
          0, 0, 1,  0,
          0, 0, 0,  1;
}

void KalmanFilter::SetQ(const float dt, const float noise_ax, const float noise_ay) {
    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt3 * dt;
    float dt4_4 = dt4 / 4;
    float dt3_2 = dt3 / 2;

    Q_ = MatrixXd(4, 4);
    Q_ << noise_ax * dt4_4, 0, noise_ax*dt3_2, 0,
          0, noise_ay*dt4_4, 0, noise_ay*dt3_2,
          noise_ax * dt3_2, 0, noise_ax*dt2, 0,
          0, noise_ay*dt3_2, 0, noise_ay*dt2;
}

void KalmanFilter::Predict() {
    /**
     * predict the state
     */
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    /**
     * update the state by using Kalman Filter equations
     */
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    SharedUpdate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    /**
     * update the state by using Extended Kalman Filter equations
     */
    VectorXd z_pred = CartesianToPolar(x_);
    VectorXd y = z - z_pred;
    // Make sure the angle is between -pi and pi
    NormalizeAngle(y(1));
    SharedUpdate(y);
}

void KalmanFilter::SharedUpdate(const VectorXd &y) {
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
