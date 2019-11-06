#ifndef AKELEY_JELLY_PHYSICS_HPP_
#define AKELEY_JELLY_PHYSICS_HPP_

#include <array>
#include <assert.h>
#include <memory>
#include <stddef.h>
#include <vector>

#include "simd_dvec3.hpp"

namespace akeley {

constexpr double gravity = 1.5;
constexpr double base_k = 1000.0;
constexpr double ambient_temperature = 20.0;
constexpr double node_mass = 1.0;
constexpr double non_floor_dampening = 0.25;
constexpr double floor_dampening = 1.2;

constexpr double edge_spring_k = base_k;
constexpr double face_spring_k = base_k * 2.0;
constexpr double inner_spring_k = base_k * 3.0;



constexpr double strength_from_temperature(double temperature)
{
    double lower = 0.3;
    double upper = 1.0;
    double s = 1.0 - 0.01 * (temperature - 25.0);
    return (s < lower) ? lower : (s > upper) ? upper : s;
}

template <typename VectorSystem, typename DerivativeFunction>
[[nodiscard]]
VectorSystem rk4_step(VectorSystem y, DerivativeFunction& f, double dt)
{
    auto k1 = f(y) * dt;
    auto k2 = f(y + k1 * 0.5) * dt;
    auto k3 = f(y + k2 * 0.5) * dt;
    auto k4 = f(y + k3) * dt;

    y += k1 * (1/6.);
    y += k2 * (2/6.);
    y += k3 * (2/6.);
    y += k4 * (1/6.);

    return y;
}

struct node
{
    simd_dvec3 position = simd_dvec3(0, 0, 0);
    simd_dvec3 velocity = simd_dvec3(0, 0, 0);
    double temperature = ambient_temperature;
    double strength = strength_from_temperature(ambient_temperature);
    double lowest_strength = strength_from_temperature(ambient_temperature);
};

struct spring_spec
{
    double k;
    double length;
};

struct spring
{
    size_t node_index_0;
    size_t node_index_1;
    spring_spec spec;
};

template <size_t GridWidth>
struct grid_coordinate
{
    size_t x;
    size_t y;
    size_t z;

    grid_coordinate(size_t x_, size_t y_, size_t z_) :
        x(x_),
        y(y_),
        z(z_)
    {
        assert(x_ <= GridWidth);
        assert(y_ <= GridWidth);
        assert(z_ <= GridWidth);
    }

    size_t to_index() const
    {
        auto gw1 = GridWidth + 1;
        return x*gw1*gw1 + y*gw1 + z;
    }
};

template <size_t GridWidth>
constexpr size_t cube_node_count =
    (GridWidth+1) * (GridWidth+1) * (GridWidth+1);

template <size_t GridWidth> class jelly_cube;

template <size_t GridWidth>
struct jelly_cube_derivative
{
    friend class jelly_cube<GridWidth>;
    static constexpr size_t node_count = cube_node_count<GridWidth>;

    std::array<simd_dvec3, node_count> position_derivative;
    std::array<simd_dvec3, node_count> velocity_derivative;

    void add_spring_velocity_derivative(const jelly_cube<GridWidth>&, spring);
    void add_cell_velocity_derivative(
        const jelly_cube<GridWidth>&,
        grid_coordinate<GridWidth>);
    void add_friction_dampening(const jelly_cube<GridWidth>&);

    jelly_cube_derivative operator* (double scale)
    {
        jelly_cube_derivative result;
        for (size_t i = 0; i < node_count; ++i) {
            result.position_derivative[i] = scale * position_derivative[i];
            result.velocity_derivative[i] = scale * velocity_derivative[i];
        }
        return result;
    }
};

template <size_t GridWidth>
class jelly_cube
{
    friend class jelly_cube_derivative<GridWidth>;
    static constexpr size_t node_count = cube_node_count<GridWidth>;

    std::array<node, node_count> nodes;
    std::shared_ptr<const std::vector<spring>> springs_ptr;
    // Note this vector is always the same for all jelly_cubes of a given
    // GridWidth; can be replaced with a constexpr std::vector in C++20.

  public:
    jelly_cube()
    {
        init_node_positions(&nodes);
        springs_ptr =
            std::make_shared<std::vector<spring>>(make_springs());
    }

    // Get rid of the move constructor to avoid null shared_ptr
    // (there's not much performance benefit of moves anyway for
    // std::array, the other member variable).
    jelly_cube(const jelly_cube&) = default;
    jelly_cube& operator= (const jelly_cube&) = default;
    ~jelly_cube() = default;

    const node& view_node(grid_coordinate<GridWidth> coord)
    {
        return nodes[coord.to_index()];
    }

    node* get_node(grid_coordinate<GridWidth> coord) {
        return const_cast<node*>(&view_node(coord));
    }

    void step_by_dt(double dt)
    {
        *this = rk4_step(*this, calculate_derivative, dt);
        enforce_floor();
    }

    jelly_cube operator+ (const jelly_cube_derivative<GridWidth>& deriv) const
    {
        jelly_cube result = *this;
        return result += deriv;
    }

    jelly_cube&
    operator+= (const jelly_cube_derivative<GridWidth>& deriv) noexcept
    {
        for (size_t i = 0; i < node_count; ++i) {
            nodes[i].position += deriv.position_derivative[i];
            nodes[i].velocity += deriv.velocity_derivative[i];
        }
        return *this;
    }

  private:
    static void init_node_positions(std::array<node, node_count>* nodes_ptr)
    {
        auto& nodes = *nodes_ptr;
        double double_grid_width = GridWidth;
        for (size_t x = 0; x <= GridWidth; ++x) {
            for (size_t y = 0; y <= GridWidth; ++y) {
                for (size_t z = 0; z <= GridWidth; ++z) {
                    grid_coordinate<GridWidth> coord(x, y, z);
                    size_t index = coord.to_index();
                    nodes[index].position = simd_dvec3(
                        double_grid_width / x,
                        double_grid_width / y,
                        double_grid_width / z);
                }
            }
        }
    }

    static std::vector<spring> make_springs()
    {
        std::vector<spring> springs;
        auto to_index = [] (size_t x, size_t y, size_t z)
        {
            return grid_coordinate<GridWidth>(x, y, z).to_index();
        };

        spring_spec horizontal_edge_spring { edge_spring_k, 1.0 / GridWidth },
            face_spring { face_spring_k, 1.4142135623730951 / GridWidth },
            inner_spring { inner_spring_k, 1.7320508075688772 / GridWidth };

        auto maybe_push_spring = [&springs, to_index]
        (size_t idx0, size_t x, size_t y, size_t z, spring_spec spec)
        {
            if (x > GridWidth || y > GridWidth || z > GridWidth) {
                return;
            }
            spring spr { idx0, to_index(x, y, z) , spec };
            springs.push_back(spr);
        };

        auto vertical_edge_spring = [horizontal_edge_spring] (size_t y)
        {
            auto force_needed = (GridWidth - y) * node_mass * gravity;
            auto displacement_needed = force_needed / edge_spring_k;
            return spring_spec {
                horizontal_edge_spring.k,
                horizontal_edge_spring.length + displacement_needed };
        };

        for (size_t x = 0; x < GridWidth; ++x) {
            for (size_t y = 0; y < GridWidth; ++y) {
                for (size_t z = 0; z < GridWidth; ++z) {
                    size_t idx0 = to_index(x, y, z);

                    maybe_push_spring(idx0, x, y, z+1, horizontal_edge_spring);
                    maybe_push_spring(idx0, x, y+1, z, vertical_edge_spring(y));
                    maybe_push_spring(idx0, x+1, y, z, horizontal_edge_spring);
                    maybe_push_spring(idx0, x+1, y+1, z, face_spring);
                    maybe_push_spring(idx0, x+1, y, z+1, face_spring);
                    maybe_push_spring(idx0, x, y+1, z+1, face_spring);
                    maybe_push_spring(idx0, x+1, y+1, z+1, inner_spring);
                }
            }
        }

        return springs;
    }

    void enforce_floor()
    {
        for (node& n : nodes) {
            auto pos = n.position;
            auto y = Y(pos);
            auto vel = n.velocity;
            auto floor_pos = simd_dvec3(X(pos), 0, Z(pos));
            auto floor_vel = simd_dvec3(X(vel), 0, Z(vel));

            n.position = step(pos, floor_pos, -y);
            n.velocity = step(vel, floor_vel, -y);
        }
    }

    static jelly_cube_derivative<GridWidth>
    calculate_derivative(const jelly_cube& arg)
    {
        jelly_cube_derivative<GridWidth> result;

        for (size_t i = 0; i < node_count; ++i) {
            result.position_derivative[i] = arg.nodes[i].velocity;
            result.velocity_derivative[i] = simd_dvec3(0, -gravity, 0);
        }

        for (const spring& the_spring : *arg.springs_ptr) {
            result.add_spring_velocity_derivative(arg, the_spring);
        }

        for (size_t x = 0; x < GridWidth; ++x) {
            for (size_t y = 0; y < GridWidth; ++y) {
                for (size_t z = 0; z < GridWidth; ++z) {
                    result.add_cell_velocity_derivative(arg, {x, y, z});
                }
            }
        }

        result.add_friction_dampening(arg);

        return result;
    }
};

template <size_t GridWidth>
void jelly_cube_derivative<GridWidth>::add_spring_velocity_derivative(
    const jelly_cube<GridWidth>& cube,
    spring the_spring)
{
    const node& node_0 = cube.nodes.at(the_spring.node_index_0);
    const node& node_1 = cube.nodes.at(the_spring.node_index_1);

    double distance;
    simd_dvec3 displacement = node_1.position - node_0.position;
    simd_dvec3 unit_displacement = normalize(displacement, &distance);
    auto kX = the_spring.spec.k * (distance - the_spring.spec.length);
    auto force = kX * node_0.strength * node_1.strength;
    auto accel = force * (1.0 / node_mass);
    simd_dvec3 added_vector = accel * unit_displacement;

    velocity_derivative[the_spring.node_index_0] += added_vector;
    velocity_derivative[the_spring.node_index_1] -= added_vector;
}

template <size_t GridWidth>
void jelly_cube_derivative<GridWidth>::add_cell_velocity_derivative(
    const jelly_cube<GridWidth>& cube,
    grid_coordinate<GridWidth> coord)
{
    // TODO
}

template <size_t GridWidth>
void jelly_cube_derivative<GridWidth>::add_friction_dampening(
    const jelly_cube<GridWidth>& cube)
{
    // TODO
    for (const node& n : cube.nodes) {
        auto vel = n.velocity;

    }
}

} // end namespace akeley

#endif /* !AKELEY_JELLY_PHYSICS_HPP_ */
