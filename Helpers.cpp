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

void helpers::extract_connected_component(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXi &C,
    const Eigen::VectorXi &K,
    const int component,
    Eigen::MatrixXd &V_component,
    Eigen::MatrixXi &F_component)
{
    // Extract the vertices from the connected component
    int n_component_vertices = K(component);
    V_component = Eigen::MatrixXd(n_component_vertices, V.cols());
    int new_vertex_index = 0;
    Eigen::VectorXi old_to_new_vertex_index = Eigen::VectorXi::Constant(V.rows(), -1);
    for (int i = 0; i < V.rows() && new_vertex_index < n_component_vertices; i++)
    {
        if (C(i) == component)
        {
            V_component.row(new_vertex_index) = V.row(i);
            old_to_new_vertex_index(i) = new_vertex_index;
            new_vertex_index++;
        }
    }

    // Use the old_to_new_vertex_index map to extract the simplices
    F_component = F;
    int new_simplex_index = 0;
    for (int i = 0; i < F.rows(); i++)
    {
        // If the first vertex of the simplex belongs to the component
        if (C(F(i, 0)) == component)
        {
            // Extract the component simplex
            for (int j = 0; j < F.cols(); j++)
                F_component(new_simplex_index, j) = old_to_new_vertex_index(F(i, j));
            new_simplex_index++;
        }
    }
    F_component = F_component.topRows(new_simplex_index);
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
            Eigen::VectorXi C, K;
            igl::connected_components(A, C, K);
            // Terminate if there is not exactly one component
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