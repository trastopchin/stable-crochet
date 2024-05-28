#include <iostream>

#include <igl/opengl/glfw/Viewer.h>
#include <imgui.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/read_triangle_mesh.h>

#include "Helpers.h"

Eigen::MatrixXd V;
Eigen::MatrixXi F;

// ARAP-based
Eigen::MatrixXd V_arap;
Eigen::MatrixXi E_arap;
Eigen::MatrixXd V_uv;
double rotation = 0.0;
double distortion = 0.0;
double max_difference = 0.0;

// igarashi
Eigen::MatrixXd V_igarashi;
Eigen::MatrixXi E_igarashi;
double stitch_width = 0.77;  // cm
double stitch_height = 0.77; // cm

// Color
auto gold = Eigen::RowVector3d(igl::GOLD_DIFFUSE[0], igl::GOLD_DIFFUSE[1], igl::GOLD_DIFFUSE[2]);

bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier)
{
    // View input mesh
    if (key == '1')
    {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        viewer.core().align_camera_center(V, F);
    }
    // View ARAP-parameterization
    else if (key == '2') {
        viewer.data().clear();
        viewer.data().set_mesh(V_uv, F);
        viewer.core().align_camera_center(V_uv, F);
    }
    // View ARAP-based
    else if (key == '3')
    {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        viewer.data().set_edges(V_arap, E_arap, gold);
        viewer.data().set_points(V_arap, gold);
        viewer.core().align_camera_center(V, F);
    }
    // View igarashi
    else if (key == '4')
    {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        viewer.data().set_edges(V_igarashi, E_igarashi, gold);
        viewer.data().set_points(V_igarashi, gold);
        viewer.core().align_camera_center(V, F);
    }
    viewer.data().compute_normals();
    return false;
}

void arap_based_helper_1(
    igl::opengl::glfw::Viewer &viewer,
    const double resampling_height,
    const double resampling_width,
    Eigen::MatrixXd &V_cat,
    Eigen::MatrixXi &E_cat)
{
    // Compute ARAP parameterization
    arap_based::parameterization(V, F, V_uv);
    std::tie(distortion, max_difference) = arap_based::distortion(V, V_uv, F);

    // Update the viewer
    key_down(viewer, '2', 0);
}

void arap_based_helper_2(
    igl::opengl::glfw::Viewer &viewer,
    const double resampling_height,
    const double resampling_width,
    Eigen::MatrixXd &V_cat,
    Eigen::MatrixXi &E_cat)
{
    // Regularly sample the parameterization
    std::vector<Eigen::MatrixXd> V_contours_resampled;
    std::vector<Eigen::MatrixXi> E_contours_resampled;
    arap_based::resampling(V, F, V_uv, resampling_height, resampling_width, rotation, V_contours_resampled, E_contours_resampled);
    
    // Add the pushed forward resampled contours
    std::tie(V_cat, E_cat) = helpers::cat_mesh(V_contours_resampled, E_contours_resampled);

    // Construct the stitch graph
    std::vector<std::vector<std::vector<int>>> A_contours;
    std::cout << "started meshing" << std::endl;
    igarashi::meshing(V_contours_resampled, E_contours_resampled, A_contours);

    // Visualize the stitch graph
    std::vector<Eigen::MatrixXd> V_mesh_list;
    std::vector<Eigen::MatrixXi> E_mesh_list;
    for (int i = 0; i < A_contours.size(); i++)
    {
        Eigen::MatrixXd V_mesh;
        igl::cat(1, V_contours_resampled[i], V_contours_resampled[i + 1], V_mesh);
        V_mesh_list.push_back(V_mesh);

        std::vector<Eigen::RowVector2i> edges;
        for (int j = 0; j < A_contours[i].size(); j++)
        {
            for (int k : A_contours[i][j])
            {
                edges.push_back(Eigen::RowVector2i(j, k));
            }
        }
        Eigen::MatrixXi E_mesh(edges.size(), 2);
        for (int j = 0; j < edges.size(); j++)
            E_mesh.row(j) = edges[j];
        E_mesh_list.push_back(E_mesh);
    }

    // Add the stitch graph
    Eigen::MatrixXd V_cat_2;
    Eigen::MatrixXi E_cat_2;
    std::tie(V_cat_2, E_cat_2) = helpers::cat_mesh(V_mesh_list, E_mesh_list);
    std::tie(V_cat, E_cat) = helpers::cat_mesh(V_cat, E_cat, V_cat_2, E_cat_2);

    // Update the viewer
    key_down(viewer, '3', 0);
}

void igarashi_helper(
    igl::opengl::glfw::Viewer &viewer,
    double contour_width,
    double sampling_width,
    Eigen::MatrixXd &V_cat,
    Eigen::MatrixXi &E_cat)
{
    auto start_seq = Eigen::seq(36, 67);

    Eigen::MatrixXd V_start = V(start_seq, Eigen::all);
    Eigen::MatrixXi E_start = helpers::ordered_edge_matrix(V_start.rows());

    std::vector<Eigen::MatrixXd> V_contours;
    std::vector<Eigen::MatrixXi> E_contours;
    std::cout << "started wrapping" << std::endl;
    igarashi::wrapping(V, F, V_start, E_start, contour_width, V_contours, E_contours);
    std::cout << "finished wrapping" << std::endl;

    std::vector<Eigen::MatrixXd> V_contours_resampled;
    std::vector<Eigen::MatrixXi> E_contours_resampled;
    std::cout << "started resampling" << std::endl;
    igarashi::resampling(V_contours, E_contours, sampling_width, V_contours_resampled, E_contours_resampled);
    std::cout << "finished resampling" << std::endl;

    // Add the resampled contours
    std::tie(V_cat, E_cat) = helpers::cat_mesh(V_contours_resampled, E_contours_resampled);

    std::vector<std::vector<std::vector<int>>> A_contours;
    std::cout << "started meshing" << std::endl;
    igarashi::meshing(V_contours_resampled, E_contours_resampled, A_contours);
    std::cout << "finished meshing" << std::endl;

    // Visualize the stitch graph
    std::vector<Eigen::MatrixXd> V_mesh_list;
    std::vector<Eigen::MatrixXi> E_mesh_list;
    for (int i = 0; i < A_contours.size(); i++)
    {
        Eigen::MatrixXd V_mesh;
        igl::cat(1, V_contours_resampled[i], V_contours_resampled[i + 1], V_mesh);
        V_mesh_list.push_back(V_mesh);

        std::vector<Eigen::RowVector2i> edges;
        for (int j = 0; j < A_contours[i].size(); j++)
        {
            for (int k : A_contours[i][j])
            {
                edges.push_back(Eigen::RowVector2i(j, k));
            }
        }
        Eigen::MatrixXi E_mesh(edges.size(), 2);
        for (int j = 0; j < edges.size(); j++)
            E_mesh.row(j) = edges[j];
        E_mesh_list.push_back(E_mesh);
    }

    // Add the stitch graph
    Eigen::MatrixXd V_cat_2;
    Eigen::MatrixXi E_cat_2;
    std::tie(V_cat_2, E_cat_2) = helpers::cat_mesh(V_mesh_list, E_mesh_list);
    std::tie(V_cat, E_cat) = helpers::cat_mesh(V_cat, E_cat, V_cat_2, E_cat_2);

    // Update the viewer
    key_down(viewer, '4', 0);
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage stable_crochet input.obj" << std::endl;
        std::exit(0);
    }

    // Viewer
    igl::opengl::glfw::Viewer viewer;

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

    // Draw the viewer menu
    menu.callback_draw_viewer_menu = [&]()
    {
        menu.draw_viewer_menu();
    };

    // Draw a custom window
    menu.callback_draw_custom_window = [&]()
    {
        // Get the size of the viewer window
        int viewer_width, viewer_height;
        glfwGetWindowSize(viewer.window, &viewer_width, &viewer_height);

        // Define next window position + size
        float window_width = 200;
        float window_height = 200;
        float window_x = viewer_width - window_width;
        ImGui::SetNextWindowPos(ImVec2(window_x, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(window_width, window_height), ImGuiCond_FirstUseEver);
        ImGui::Begin(
            "Parameters", nullptr,
            ImGuiWindowFlags_NoSavedSettings);

        // Parameters
        ImGui::PushItemWidth(-80);
        ImGui::InputDouble("stitch_width", &stitch_width);
        ImGui::InputDouble("stitch_height", &stitch_height);

        if (ImGui::Button("ARAP-parameterization"))
            arap_based_helper_1(viewer, stitch_height, stitch_width, V_arap, E_arap);
            if (ImGui::Button("ARAP-based"))
            arap_based_helper_2(viewer, stitch_height, stitch_width, V_arap, E_arap);
        ImGui::InputDouble("rotation", &rotation);
        ImGui::Text("distortion: %lf", distortion);
        ImGui::Text("max differ: %lf", max_difference);

        if (ImGui::Button("igarashi"))
            igarashi_helper(viewer, stitch_height, stitch_width, V_igarashi, E_igarashi);

        ImGui::PopItemWidth();

        ImGui::End();
    };

    // Read the triangle mesh
    igl::read_triangle_mesh(argv[1], V, F);

    // Plot the mesh
    viewer.data().set_mesh(V, F);
    viewer.callback_key_down = &key_down;
    
    // Make the label size a bit bigger
    viewer.data().label_size = 2;

    // Launch the viewer
    viewer.launch();
}