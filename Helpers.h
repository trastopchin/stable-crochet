#include <vector>
#include <algorithm>
#include <set>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/cat.h>
#include <igl/arap.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/edge_lengths.h>
#include <igl/barycentric_coordinates.h>
#include <igl/barycentric_interpolation.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/isolines.h>
#include <igl/adjacency_matrix.h>
#include <igl/adjacency_list.h>
#include <igl/connected_components.h>

namespace helpers
{
    typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayXb;

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> cat_mesh(
        const Eigen::MatrixXd &V1,
        const Eigen::MatrixXi &F1,
        const Eigen::MatrixXd &V2,
        const Eigen::MatrixXi &F2);

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> cat_mesh(
        std::vector<Eigen::MatrixXd> &V_list,
        std::vector<Eigen::MatrixXi> &F_list);

    void covariance_analysis(
        const Eigen::MatrixXd& P,
        Eigen::MatrixXd& eigenvectors, Eigen::VectorXd& eigenvalues
    );

    Eigen::MatrixXi ordered_edge_matrix(
        int n_vertices);

    Eigen::MatrixXd upsample_curve(
        const Eigen::MatrixXd &V,
        const double interval);

    Eigen::VectorXi vertex_indices(
        int n_vertices);
}

namespace arap_based
{
    void parameterization(
        const Eigen::MatrixXd &V,
        const Eigen::MatrixXi &F,
        Eigen::MatrixXd &V_uv);

    std::tuple<double, double> distortion(
        const Eigen::MatrixXd &V1,
        const Eigen::MatrixXd &V2,
        const Eigen::MatrixXi &F);

    void resampling(
        const Eigen::MatrixXd &V,
        const Eigen::MatrixXi &F,
        const Eigen::MatrixXd &V_uv,
        const double resampling_height,
        const double resampling_width,
        const double rotation,
        std::vector<Eigen::MatrixXd> &V_contours_resampled,
        std::vector<Eigen::MatrixXi> &E_contours_resampled);

    void P_uv_to_B_and_F(
        const Eigen::MatrixXd &V_uv,
        const Eigen::MatrixXi &F,
        const Eigen::MatrixXd &P_uv,
        Eigen::MatrixXd &P_uv_to_B,
        Eigen::VectorXi &P_uv_to_F);

    void push_points_forward(
        const Eigen::MatrixXd &V,
        const Eigen::MatrixXi &F,
        const Eigen::MatrixXd &V_uv,
        const Eigen::MatrixXd &P_uv,
        Eigen::MatrixXd &P);
}

namespace igarashi
{
    void wrapping(
        const Eigen::MatrixXd &V,
        const Eigen::MatrixXi &F,
        const Eigen::MatrixXd &V_start,
        const Eigen::MatrixXi &E_start,
        const double contour_width,
        std::vector<Eigen::MatrixXd> &V_contours,
        std::vector<Eigen::MatrixXi> &E_contours);

    void resampling(
        const std::vector<Eigen::MatrixXd> &V_contours,
        const std::vector<Eigen::MatrixXi> &E_contours,
        const double sampling_width,
        std::vector<Eigen::MatrixXd> &V_contours_resampled,
        std::vector<Eigen::MatrixXi> &E_contours_resampled);

    void meshing(
        const std::vector<Eigen::MatrixXd> &V_contours_resampled,
        const std::vector<Eigen::MatrixXi> &E_contours_resampled,
        std::vector<std::vector<std::vector<int>>> &A_contours);

    void pattern_generation(
        const std::vector<std::vector<std::vector<int>>> &A,
        std::string pattern);
}