#include "voxel_map.h"

namespace lio
{

    UnionFindNode::UnionFindNode(double _plane_thresh, int _update_size_thresh, int _max_point_thresh, Eigen::Vector3d &_voxel_center, VoxelMap *_map)
        : plane_thesh(_plane_thresh),
          update_size_thresh(_update_size_thresh),
          max_point_thresh(_max_point_thresh),
          voxel_center(_voxel_center),
          map(_map)
    {
        newly_added_num = 0;
        update_enable = true;
        plane = std::make_shared<Plane>();
        root_node = this;
        temp_points.clear();
    }

    void UnionFindNode::push(const PointWithCov &pv)
    {
        addToPlane(pv);
        temp_points.push_back(pv);
    }

    void UnionFindNode::emplace(const PointWithCov &pv)
    {
        if (!isInitialized())
        {
            addToPlane(pv);
            temp_points.push_back(pv);
            newly_added_num += 1;
            if (newly_added_num >= update_size_thresh)
            {
                updatePlane();
                newly_added_num = 0;
            }
        }
        else
        {
            if (isPlane())
            {
                if (update_enable)
                {
                    addToPlane(pv);
                    temp_points.push_back(pv);
                    newly_added_num++;
                    if (newly_added_num >= update_size_thresh)
                    {
                        updatePlane();
                        newly_added_num = 0;
                    }
                    if (temp_points.size() > max_point_thresh)
                    {
                        update_enable = false;
                        std::vector<PointWithCov>().swap(temp_points);
                    }
                }
                else
                {
                    // try merge
                }
            }
            else
            {
                if (update_enable)
                {
                    addToPlane(pv);
                    temp_points.push_back(pv);
                    newly_added_num++;
                    if (newly_added_num >= update_size_thresh)
                    {
                        updatePlane();
                        newly_added_num = 0;
                    }
                    if (temp_points.size() > max_point_thresh)
                    {
                        update_enable = false;
                        std::vector<PointWithCov>().swap(temp_points);
                    }
                }
            }
        }
    }

    void UnionFindNode::addToPlane(const PointWithCov &pv)
    {

        plane->mean = plane->mean + (pv.point - plane->mean) / (plane->n + 1.0);
        plane->ppt += pv.point * pv.point.transpose();
        plane->n += 1;
    }

    void UnionFindNode::updatePlane()
    {
        assert(temp_points.size() == plane->n);
        if (plane->n < update_size_thresh)
            return;
        plane->is_init = true;
        Eigen::Matrix3d cov = plane->ppt / static_cast<double>(plane->n) - plane->mean * plane->mean.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
        Eigen::Matrix3d evecs = es.eigenvectors();
        Eigen::Vector3d evals = es.eigenvalues();

        if (evals(0) > plane_thesh)
        {
            plane->is_plane = false;
            return;
        }
        plane->is_plane = true;
        plane->norm = evecs.col(0);
        Eigen::Matrix3d J_Q = Eigen::Matrix3d::Identity() / static_cast<double>(plane->n);

        for (PointWithCov &pv : temp_points)
        {
            Eigen::Matrix<double, 6, 3> J;
            Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
            for (int m = 1; m < 3; m++)
            {
                Eigen::Matrix<double, 1, 3> F_m = (pv.point - plane->mean).transpose() / ((plane->n) * (evals(0) - evals(m))) *
                                                  (evecs.col(m) * evecs.col(0).transpose() +
                                                   evecs.col(0) * evecs.col(m).transpose());
                F.row(m) = F_m;
            }
            J.block<3, 3>(0, 0) = evecs * F;
            J.block<3, 3>(3, 0) = J_Q;
            plane->plane_cov += J * pv.cov * J.transpose();
        }
        if (plane->mean.dot(plane->norm) < 0)
            plane->norm = -plane->norm;
        plane->axis_distance = plane->mean.dot(plane->norm);
    }

    VoxelMap::VoxelMap(double _voxel_size, double _plane_thresh, int _update_size_thresh, int _max_point_thresh)
        : voxel_size(_voxel_size),
          plane_thresh(_plane_thresh),
          update_size_thresh(_update_size_thresh),
          max_point_thresh(_max_point_thresh)
    {
    }

    VoxelKey VoxelMap::index(const Eigen::Vector3d &point)
    {
        Eigen::Vector3d idx = (point / voxel_size).array().floor();
        return VoxelKey(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2)));
    }

    void VoxelMap::build(std::vector<PointWithCov> &pvs)
    {
        for (PointWithCov &pv : pvs)
        {
            VoxelKey k = index(pv.point);
            auto f_it = feat_map.find(k);
            if (f_it == feat_map.end())
            {
                Eigen::Vector3d center(static_cast<double>(k.x) + 0.5, static_cast<double>(k.y) + 0.5, static_cast<double>(k.z) + 0.5);
                center *= voxel_size;
                feat_map[k] = new UnionFindNode(plane_thresh, update_size_thresh, max_point_thresh, center, this);
            }
            feat_map[k]->push(pv);
        }
        for (auto it = feat_map.begin(); it != feat_map.end(); it++)
        {
            it->second->updatePlane();
        }
    }

    void VoxelMap::update(std::vector<PointWithCov> &pvs)
    {
        for (PointWithCov &pv : pvs)
        {
            VoxelKey k = index(pv.point);
            auto f_it = feat_map.find(k);
            if (f_it == feat_map.end())
            {
                Eigen::Vector3d center(static_cast<double>(k.x) + 0.5, static_cast<double>(k.y) + 0.5, static_cast<double>(k.z) + 0.5);
                center *= voxel_size;
                feat_map[k] = new UnionFindNode(plane_thresh, update_size_thresh, max_point_thresh, center, this);
            }
            feat_map[k]->emplace(pv);
        }
    }

    void VoxelMap::buildResidual(ResidualData &info, UnionFindNode *node)
    {
        info.is_valid = false;
        if (node->isPlane())
        {
            Eigen::Vector3d p_world_to_center = info.point_world - node->plane->mean;
            info.plane_mean = node->plane->mean;
            info.plane_cov = node->plane->plane_cov;
            info.plane_norm = node->plane->norm;
            
            info.residual = info.plane_norm.transpose() * p_world_to_center;
            
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = p_world_to_center;
            J_nq.block<1, 3>(0, 3) = -info.plane_norm;

            double sigma_l = J_nq * info.plane_cov * J_nq.transpose();
            sigma_l += info.plane_norm.transpose() * info.cov_world * info.plane_norm;

            if (std::abs(info.residual) < 3.0 * sqrt(sigma_l))
                info.is_valid = true;
            else
                info.is_valid = false;
        }
    }

    VoxelMap::~VoxelMap()
    {
        if (feat_map.size() == 0)
            return;
        for (auto it = feat_map.begin(); it != feat_map.end(); it++)
        {
            delete it->second;
        }
        FeatMap().swap(feat_map);
    }
} // namespace lio
