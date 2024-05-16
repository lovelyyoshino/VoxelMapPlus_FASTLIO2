#include "commons.h"
namespace lio
{

    Eigen::Vector3d rotate2rpy(Eigen::Matrix3d &rot)
    {
        double roll = std::atan2(rot(2, 1), rot(2, 2));
        double pitch = asin(-rot(2, 0));
        double yaw = std::atan2(rot(1, 0), rot(0, 0));
        return Eigen::Vector3d(roll, pitch, yaw);
    }

    float sq_dist(const pcl::PointXYZINormal &p1, const pcl::PointXYZINormal &p2)
    {
        return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    }

    //https://github.com/hku-mars/VoxelMap/blob/master/include/voxel_map_util.hpp#L1222
    void calcBodyCov(Eigen::Vector3d &pb, const double &range_inc, const double &degree_inc, Eigen::Matrix3d &cov)
    {
        if (pb[2] == 0)
        {
            pb[2] = 0.001;
        }
        double range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);//求相对于原点的距离
        double range_var = range_inc * range_inc;
        Eigen::Matrix2d direction_var;
        direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
            pow(sin(DEG2RAD(degree_inc)), 2);//角度的方差
        Eigen::Vector3d direction(pb);
        direction.normalize();//取单位向量
        Eigen::Matrix3d direction_hat;
        direction_hat << 0, -direction(2), direction(1), direction(2), 0,
            -direction(0), -direction(1), direction(0), 0;//对方向取反对称矩阵
        //计算两个基向量 base_vector1 和 base_vector2，这些基向量与方向向量 direction 正交。协方差矩阵第二大特征值对应的特征向量v_2，其实就是这里数据 X^{*}对应的协方差矩阵，特征值最大的特征向量w。其方向就是与特征向量v1垂直方向上，数据方差最大的方向
        Eigen::Vector3d base_vector1(1, 1,
                                     -(direction(0) + direction(1)) / direction(2));
        base_vector1.normalize();
        Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
        base_vector2.normalize();
        Eigen::Matrix<double, 3, 2> N;
        N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
            base_vector1(2), base_vector2(2);//通过基于方向向量的正交基向量构建矩阵 N，这个N是为了计算协方差矩阵
        Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;//计算 A 矩阵,对应公式是 A = r * hat(d) * N
        cov = direction * range_var * direction.transpose() +
              A * direction_var * A.transpose();//计算协方差矩阵
    }
}
