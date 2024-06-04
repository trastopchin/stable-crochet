#include "Helpers.h"

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> helpers::cat_mesh(
    const Eigen::MatrixXd &V1,
    const Eigen::MatrixXi &F1,
    const Eigen::MatrixXd &V2,
    const Eigen::MatrixXi &F2)
{
    Eigen::MatrixXd V3;
    Eigen::MatrixXi F3;
    V3 = igl::cat(1, V1, V2);
    Eigen::MatrixXi F2_offset = F2 + Eigen::MatrixXi::Constant(F2.rows(), F2.cols(), V1.rows());
    F3 = igl::cat(1, F1, F2_offset);
    return std::make_tuple(V3, F3);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> helpers::cat_mesh(
    std::vector<Eigen::MatrixXd> &V_list,
    std::vector<Eigen::MatrixXi> &F_list)
{
    // Compute the total number of rows
    int n_vertices = 0;
    int n_simplices = 0;
    int n_meshes = V_list.size();
    for (int i = 0; i < n_meshes; i++)
    {
        n_vertices += V_list[i].rows();
        n_simplices += F_list[i].rows();
    }

    Eigen::MatrixXd V_cat(n_vertices, V_list[0].cols());
    Eigen::MatrixXi F_cat(n_simplices, F_list[0].cols());

    // Concatenate the vertex and simplex arrays
    int vertex_index = 0;
    int simplex_index = 0;
    for (int i = 0; i < n_meshes; i++)
    {
        // Concatenate the vertex array
        Eigen::MatrixXd &V = V_list[i];
        V_cat.middleRows(vertex_index, V.rows()) = V;

        // Offset and concatenate the simplex array
        Eigen::MatrixXi &F = F_list[i];
        Eigen::MatrixXi offset = Eigen::MatrixXi::Constant(F.rows(), F.cols(), vertex_index);
        F_cat.middleRows(simplex_index, F.rows()) = F + offset;

        // Update the vertex and simplex index
        vertex_index += V.rows();
        simplex_index += F.rows();
    }

    return std::make_tuple(V_cat, F_cat);
}

// https://www.graphics.rwth-aachen.de/media/papers/p_Pau021.pdf
void helpers::covariance_analysis(
    const Eigen::MatrixXd& P,
    Eigen::MatrixXd& eigenvectors, Eigen::VectorXd& eigenvalues
)
{
    // Construct the 3 x 3 covariance matrix C
    auto mean = P.colwise().mean();
    auto difference = P.rowwise() - mean;
    auto C = difference.transpose() * difference;

    // Eigen analysis
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(C);
    eigenvectors = eigen_solver.eigenvectors();
    eigenvalues = eigen_solver.eigenvalues();
}

Eigen::MatrixXi helpers::ordered_edge_matrix(int n_vertices)
{
    Eigen::MatrixXi E = Eigen::MatrixXi(n_vertices - 1, 2);
    for (int i = 0; i < n_vertices - 1; i++)
        E.row(i) = Eigen::RowVector2i(i, i + 1);
    return E;
}

Eigen::MatrixXd helpers::upsample_curve(
    const Eigen::MatrixXd &V,
    const double interval)
{
    // Estimate the number of upsampled points
    double curve_length = 0;
    for (int i = 0; i < V.rows() - 1; i++)
        curve_length += (V.row(i + 1) - V.row(i)).norm();
    std::vector<Eigen::MatrixXd> V_upsampled_list;
    V_upsampled_list.reserve(ceil(curve_length / interval));

    // Upsample the curve
    for (int i = 0; i < V.rows() - 1; i++)
    {
        Eigen::RowVectorXd p1 = V.row(i);
        Eigen::RowVectorXd p2 = V.row(i + 1);
        double d = (p2 - p1).norm();
        int num_samples = ceil(d / interval);
        for (int j = 0; j < num_samples; j++)
        {
            double t = (double)j / num_samples;
            Eigen::RowVectorXd p_sampled = (1 - t) * p1 + t * p2;
            V_upsampled_list.push_back(p_sampled);
        }
    }
    V_upsampled_list.push_back(V.row(V.rows() - 1));

    // Return the vertex matrix
    Eigen::MatrixXd V_upsampled;
    igl::cat(1, V_upsampled_list, V_upsampled);
    return V_upsampled;
}

Eigen::VectorXi helpers::vertex_indices(
    int n_vertices)
{
    Eigen::VectorXi vertex_indices(n_vertices);
    for (int i = 0; i < n_vertices; i++)
        vertex_indices(i) = i;
    return vertex_indices;
}

void arap_based::parameterization(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    Eigen::MatrixXd &V_uv)
{
    // Compute the initial solution for ARAP (harmonic parametrization)
    Eigen::VectorXi bnd;
    igl::boundary_loop(F, bnd);
    Eigen::MatrixXd bnd_uv;
    igl::map_vertices_to_circle(V, bnd, bnd_uv);
    Eigen::MatrixXd initial_guess;
    igl::harmonic(V, F, bnd, bnd_uv, 1, initial_guess);

    // Initialize ARAP
    igl::ARAPData arap_data;
    arap_data.max_iter = 100;

    // Fixed vertices
    Eigen::VectorXi b = Eigen::VectorXi::Zero(0);

    // 2 means that we're going to *solve* in 2d
    arap_precomputation(V, F, 2, b, arap_data);

    // Solve arap using the harmonic map as initial guess
    V_uv = initial_guess;

    // Boundary conditions
    Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(0, 0);
    arap_solve(bc, arap_data, V_uv);
}

std::tuple<double, double> arap_based::distortion(
    const Eigen::MatrixXd &V1,
    const Eigen::MatrixXd &V2,
    const Eigen::MatrixXi &F)
{
    Eigen::MatrixXd L1, L2;
    igl::edge_lengths(V1, F, L1);
    igl::edge_lengths(V2, F, L2);

    double distortion = 0;
    double max_difference = 0;
    for (int i = 0; i < L1.rows(); i++)
        for (int j = 0; j < L2.cols(); j++)
        {
            double difference = fabs(L1(i, j) - L2(i, j));
            distortion += difference;
            max_difference = std::max(difference, max_difference);
        }

    return std::make_tuple(distortion, max_difference);
}

void arap_based::resampling(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &V_uv,
    const double resampling_height,
    const double resampling_width,
    const double rotation,
    std::vector<Eigen::MatrixXd> &V_contours_resampled,
    std::vector<Eigen::MatrixXi> &E_contours_resampled)
{
    // Clear the contours
    V_contours_resampled.clear();
    E_contours_resampled.clear();

    // Covariance analysis to "optimally" align the points for regular resampling
    Eigen::MatrixXd eigenvectors;
    Eigen::VectorXd eigenvalues;
    helpers::covariance_analysis(V_uv, eigenvectors, eigenvalues);
    Eigen::MatrixXd inverse = eigenvectors.inverse();
    Eigen::MatrixXd V_uv_aligned = V_uv * inverse.transpose();

    // Translate to center
    Eigen::RowVectorXd center = V_uv_aligned.colwise().mean();
    V_uv_aligned = V_uv_aligned.rowwise() - center;

    // Rotate
    double radians = rotation * 2 * M_PI;
    Eigen::Matrix2d rotation_matrix;
    rotation_matrix << std::cos(radians), -std::sin(radians),
                       std::sin(radians),  std::cos(radians);
    Eigen::Matrix2d rotation_matrix_inverse = rotation_matrix.inverse();
    V_uv_aligned = V_uv_aligned * rotation_matrix.transpose();

    // Compute the bounding box
    Eigen::RowVector2d bb_min, bb_max, diagonal;
    bb_min = V_uv_aligned.colwise().minCoeff();
    bb_max = V_uv_aligned.colwise().maxCoeff();
    diagonal = bb_max - bb_min;

    // Regular resampling
    double uv_width = diagonal[0];
    double uv_height = diagonal[1];
    int n_height_samples = uv_height / resampling_height;
    int n_width_samples = uv_width / resampling_width;
    std::vector<Eigen::RowVector2d> samples;
    
    bool found_first_row = false;
    for (int i = 0; i < n_height_samples; i++) {
        // Sample the row
        Eigen::MatrixXd V_contour_resampled(n_width_samples, 2);
        for (int j = 0; j < n_width_samples; j++) {
            Eigen::RowVector2d sample = bb_min + Eigen::RowVector2d(i * resampling_width, 0) + Eigen::RowVector2d(0, j * resampling_height);
            V_contour_resampled.row(j) = sample;
        }
        // Undo the rotation
        V_contour_resampled *= rotation_matrix_inverse.transpose();

        // Undo the translation
        V_contour_resampled = V_contour_resampled.rowwise() + center;

        // Undo the optimal alignment transformation
        V_contour_resampled *= eigenvectors.transpose();

        // Push the row forward
        Eigen::MatrixXd P;
        arap_based::push_points_forward(V, F, V_uv, V_contour_resampled, P);
        
        if (P.rows() != 0) {
            found_first_row = true;
            V_contour_resampled = P;
            V_contours_resampled.push_back(V_contour_resampled);
            E_contours_resampled.push_back(helpers::ordered_edge_matrix(P.rows()));
        }
        else if (P.rows() == 0 && !found_first_row)
            continue;
        else if (P.rows() == 0 && found_first_row)
            return;
    }
}

void arap_based::P_uv_to_B_and_F(
    const Eigen::MatrixXd &V_uv,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &P_uv,
    Eigen::MatrixXd &P_uv_to_B,
    Eigen::VectorXi &P_uv_to_F)
{
    // Map from each row of P_uv to its corresponding triangle
    P_uv_to_F = Eigen::VectorXi::Constant(P_uv.rows(), 1, -1);
    // Map from each row of P_uv to its corresponding barycentric coordinates
    P_uv_to_B = Eigen::MatrixXd(P_uv.rows(), 3);

    // Triangle corner points
    Eigen::MatrixXd A = V_uv(F.col(0), Eigen::all);
    Eigen::MatrixXd B = V_uv(F.col(1), Eigen::all);
    Eigen::MatrixXd C = V_uv(F.col(2), Eigen::all);
    
    for (int i = 0; i < P_uv.rows(); i++) {
        // Replicate the uv point F.rows() times
        // to get its barycentric coordinates for each triangle
        Eigen::MatrixXd P_uv_i = P_uv.row(i).replicate(F.rows(), 1);
        Eigen::MatrixXd L;
        igl::barycentric_coordinates(P_uv_i, A, B, C, L);

        // Determine which triangle the point belongs to
        double epsilon = 1e-4;
        helpers::ArrayXb sum_to_one = (L.array().rowwise().sum() - 1).abs() < epsilon;
        helpers::ArrayXb all_non_negative = (L.array() >= 0).rowwise().all();
        helpers::ArrayXb valid = sum_to_one && all_non_negative;
        for (int j = 0; j < F.rows(); j++) {
            if (valid(j)) {
                P_uv_to_F(i) = j;
                P_uv_to_B.row(i) = L.row(j);
                break;
            }
        }
    }
}

void arap_based::push_points_forward(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &V_uv,
    const Eigen::MatrixXd &P_uv,
    Eigen::MatrixXd &P)
{
    // Get the maps from uv points to barycentric coordinates and triangle indices
    Eigen::MatrixXd P_uv_to_B;
    Eigen::VectorXi P_uv_to_F;
    arap_based::P_uv_to_B_and_F(V_uv, F, P_uv, P_uv_to_B, P_uv_to_F);

    // Extract the submatrix of valid barycentric coordinates and triangle indices
    std::vector<Eigen::RowVector3d> B_valid;
    std::vector<Eigen::RowVectorXi> I_valid;
    for (int i = 0; i < P_uv_to_B.rows(); i++) {
        // If the uv point has a corresponding triangle
        if (P_uv_to_F(i) != -1) {
            B_valid.push_back(P_uv_to_B.row(i));
            I_valid.push_back(Eigen::RowVectorXi::Constant(1, 1, P_uv_to_F(i)));
        }
    }
    // Return empty P if there are no valid points
    int n_valid = B_valid.size();
    if (n_valid == 0) {
        P = Eigen::MatrixXd::Zero(0, 0);
        return;
    }
    // Otherwise push the points forward
    Eigen::MatrixXd B;
    Eigen::VectorXi I;
    igl::cat(1, B_valid, B);
    igl::cat(1, I_valid, I);
    igl::barycentric_interpolation(V, F, B, I, P);
}

void igarashi::wrapping(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &V_start,
    const Eigen::MatrixXi &E_start,
    const double contour_width,
    std::vector<Eigen::MatrixXd> &V_contours,
    std::vector<Eigen::MatrixXi> &E_contours)
{
    // Initialize the mesh of concatenated contours
    Eigen::MatrixXd V_contours_cat = V_start;
    Eigen::MatrixXi E_contours_cat = E_start;

    // Initialize the vector of contours
    V_contours.clear();
    E_contours.clear();
    V_contours.push_back(V_start);
    E_contours.push_back(E_start);

    while (true)
    {
        // Compute the Euclidean distance from the mesh to the current contours
        Eigen::VectorXd D;
        {
            Eigen::VectorXd sqrD;
            Eigen::VectorXi I;
            Eigen::MatrixXd C;
            igl::point_mesh_squared_distance(V, V_contours_cat, E_contours_cat, sqrD, I, C);
            D = sqrD.array().sqrt();
        }

        // Compute the contour_width distance away contours
        Eigen::MatrixXd V_next;
        Eigen::MatrixXi E_next;
        {
            // Get the contour_width valued isolines of the euclidean distance scalar field
            Eigen::VectorXd vals = Eigen::VectorXd::Constant(1, contour_width);
            Eigen::VectorXi I;
            igl::isolines(V, F, D, vals, V_next, E_next, I);
            // Terminate if there are no isolines
            if (V_next.rows() == 0)
                return;

            // Get the connected components of the contour_width values isolines
            Eigen::SparseMatrix<int> A;
            igl::adjacency_matrix(E_next, A);
            
            // Terminate if the contour doesn't have two boundary vertices
            Eigen::VectorXi vertex_degrees = A * Eigen::VectorXi::Ones(A.cols());
            int n_boundary_vertices = 0;
            for (int i = 0; i < vertex_degrees.rows(); i++)
                if (vertex_degrees(i) == 1)
                    n_boundary_vertices++;
            if (n_boundary_vertices != 2)
                return;

            // Terminate if there is not exactly one component
            Eigen::VectorXi C, K;
            igl::connected_components(A, C, K);
            if (K.rows() != 1)
                return;
        }

        // Concatenate the next contour
        std::tie(V_contours_cat, E_contours_cat) = helpers::cat_mesh(V_contours_cat, E_contours_cat, V_next, E_next);

        // Push back the next contour
        V_contours.push_back(V_next);
        E_contours.push_back(E_next);
    }
}

void igarashi::resampling(
    const std::vector<Eigen::MatrixXd> &V_contours,
    const std::vector<Eigen::MatrixXi> &E_contours,
    const double sampling_width,
    std::vector<Eigen::MatrixXd> &V_contours_resampled,
    std::vector<Eigen::MatrixXi> &E_contours_resampled)
{
    // Initialize the resampled contours
    int n_contours = V_contours.size();
    V_contours_resampled.clear();
    E_contours_resampled.clear();
    V_contours_resampled.reserve(n_contours);
    E_contours_resampled.reserve(n_contours);

    // Identify the starting vertex of each contour
    Eigen::VectorXi starting_vertex_indices = Eigen::VectorXi(n_contours);
    starting_vertex_indices(0) = 0; // Known for the first contour
    for (int i = 1; i < n_contours; i++)
    {
        Eigen::RowVectorXd prev_starting_vertex = V_contours[i - 1].row(starting_vertex_indices(i - 1));

        // Find the boundary vertices of the contour
        Eigen::SparseMatrix<int> A;
        igl::adjacency_matrix(E_contours[i], A);
        Eigen::VectorXi vertex_degrees = A * Eigen::VectorXi::Ones(A.cols());
        std::vector<int> boundary_vertex_indices;
        for (int i = 0; i < vertex_degrees.rows(); i++)
        {
            if (vertex_degrees(i) == 1)
                boundary_vertex_indices.push_back(i);
        }

        // Find the closest boundary vertex
        int starting_index;
        double max_dist = std::numeric_limits<double>::max();
        for (int j = 0; j < boundary_vertex_indices.size(); j++)
        {
            int boundary_vertex_index = boundary_vertex_indices[j];
            Eigen::RowVectorXd boundary_vertex = V_contours[i].row(boundary_vertex_index);
            double dist = (prev_starting_vertex - boundary_vertex).squaredNorm();
            if (dist < max_dist)
            {
                max_dist = dist;
                starting_index = boundary_vertex_index;
            }
        }
        starting_vertex_indices(i) = starting_index;
    }

    // Reorder the vertices of the contours for resampling
    std::vector<Eigen::MatrixXd> V_contours_reordered;
    V_contours_reordered.reserve(n_contours);
    for (int i = 0; i < n_contours; i++)
    {
        int n_vertices = V_contours[i].rows();

        // Reorder the first vertex
        Eigen::VectorXi old_to_new_vertex_index = Eigen::VectorXi::Constant(n_vertices, -1);
        int old_vertex_index_curr = starting_vertex_indices(i);
        old_to_new_vertex_index(old_vertex_index_curr) = 0;

        // Reorder the second vertex
        std::vector<std::vector<int>> A;
        igl::adjacency_list(E_contours[i], A);
        int old_vertex_index_prev = old_vertex_index_curr;
        old_vertex_index_curr = A[old_vertex_index_curr][0];
        old_to_new_vertex_index(old_vertex_index_curr) = 1;

        // Reorder the remaining vertices
        int new_vertex_index = 2;
        while (A[old_vertex_index_curr].size() != 1)
        {
            for (int j : A[old_vertex_index_curr])
            {
                if (j != old_vertex_index_prev)
                {
                    old_vertex_index_prev = old_vertex_index_curr;
                    old_vertex_index_curr = j;
                    old_to_new_vertex_index(old_vertex_index_curr) = new_vertex_index;
                    new_vertex_index++;
                    break;
                }
            }
        }

        // Construct the reordered vertex matrix
        Eigen::MatrixXd V_contour_reordered(n_vertices, V_contours[i].cols());
        for (int j = 0; j < n_vertices; j++)
            V_contour_reordered.row(old_to_new_vertex_index(j)) = V_contours[i].row(j);
        V_contours_reordered.push_back(V_contour_reordered);
    }

    // Resampled the reordered vertices
    for (int i = 0; i < n_contours; i++)
    {
        double upsample_width = sampling_width / 100.0;
        Eigen::MatrixXd V_contour_upsampled = helpers::upsample_curve(V_contours_reordered[i], upsample_width);
        std::vector<Eigen::RowVectorXd> V_contour_resampled_list;
        V_contour_resampled_list.reserve(V_contour_upsampled.rows());

        int index_curr = 0;
        int index_next = 0;
        V_contour_resampled_list.push_back(V_contour_upsampled.row(0));
        while (true)
        {
            while (index_next < V_contour_upsampled.rows() && (V_contour_upsampled.row(index_curr) - V_contour_upsampled.row(index_next)).norm() < sampling_width)
                index_next++;
            if (index_next >= V_contour_upsampled.rows())
                break;
            V_contour_resampled_list.push_back(V_contour_upsampled.row(index_next));
            index_curr = index_next;
        }

        Eigen::MatrixXd V_contour_resampled;
        igl::cat(1, V_contour_resampled_list, V_contour_resampled);
        V_contours_resampled.push_back(V_contour_resampled);
        E_contours_resampled.push_back(helpers::ordered_edge_matrix(V_contour_resampled.rows()));
    }
}

void igarashi::meshing(
    const std::vector<Eigen::MatrixXd> &V_contours_resampled,
    const std::vector<Eigen::MatrixXi> &E_contours_resampled,
    std::vector<std::vector<std::vector<int>>> &A_contours)
{
    int n_contours = V_contours_resampled.size();
    A_contours.clear();
    A_contours.reserve(n_contours);

    for (int i = 0; i < n_contours - 1; i++)
    {
        const Eigen::MatrixXd &V1 = V_contours_resampled[i];
        const Eigen::MatrixXd &V2 = V_contours_resampled[i + 1];
        std::vector<std::set<int>> A_contour_temp(V1.rows(), std::set<int>());

        // Compute the forward edges
        {
            Eigen::VectorXi V2_indices = helpers::vertex_indices(V2.rows());
            Eigen::VectorXd sqrD;
            Eigen::VectorXi I;
            Eigen::MatrixXd C;
            igl::point_mesh_squared_distance(V1, V2, V2_indices, sqrD, I, C);
            for (int i = 0; i < V1.rows(); i++)
            {
                int V1_index = i;
                int V2_index = I(i) + V1.rows();
                A_contour_temp[V1_index].insert(V2_index);
            }
        }

        // Compute the backward edges
        {
            Eigen::VectorXi V1_indices = helpers::vertex_indices(V1.rows());
            Eigen::VectorXd sqrD;
            Eigen::VectorXi I;
            Eigen::MatrixXd C;
            igl::point_mesh_squared_distance(V2, V1, V1_indices, sqrD, I, C);
            for (int i = 0; i < V2.rows(); i++)
            {
                int V2_index = i + V1.rows();
                int V1_index = I(i);
                A_contour_temp[V1_index].insert(V2_index);
            }
        }

        // Convert the edges into an adjacency list
        std::vector<std::vector<int>> A_contour(V1.rows(), std::vector<int>());
        for (int i = 0; i < V1.rows(); i++)
            for (int j : A_contour_temp[i])
                A_contour[i].push_back(j);
        A_contours.push_back(A_contour);
    }
}