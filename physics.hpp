#ifndef AKELEY_JELLY_PHYSICS_HPP_
#define AKELEY_JELLY_PHYSICS_HPP_

#include <array>
#include <assert.h>
#include <memory>
#include <stddef.h>
#include <vector>

#include "simd_dvec3.hpp"

namespace akeley {

constexpr double gravity = -1.5;

template <typename VectorSystem, typename DerivativeFunction>
void rk4_step(VectorSystem* y, DerivativeFunction& f, double dt)
{
    auto k1 = f(*y) * dt;
    auto k2 = f(*y + k1 * 0.5) * dt;
    auto k3 = f(*y + k2 * 0.5) * dt;
    auto k4 = f(*y + k3) * dt;

    *y += k1 * (1/6.);
    *y += k2 * (2/6.);
    *y += k3 * (2/6.);
    *y += k4 * (1/6.);
}

struct node
{
    simd_dvec3 position;
    simd_dvec3 velocity;
    double temperature;
    double strength;
    double lowest_strength;
    double mass;
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

  public:
    jelly_cube()
    {
        init_nodes(&nodes);
        springs_ptr =
            std::make_shared<std::vector<spring>>(make_springs(&nodes));
    }

    const node& view_node(grid_coordinate<GridWidth> coord)
    {
        return nodes[coord.to_index()];
    }

    node* get_node(grid_coordinate<GridWidth> coord) {
        return const_cast<node*>(&view_node(coord));
    }

    void step(double dt)
    {
        rk4_step(this, calculate_derivative, dt);
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
    static void init_nodes(std::array<node, node_count>* nodes)
    {
        // TODO
    }

    static std::vector<spring> make_springs(
        const std::array<node, node_count>* nodes)
    {
        std::vector<spring> springs;
        // TODO
        return springs;
    }

    static jelly_cube_derivative<GridWidth>
    calculate_derivative(const jelly_cube& arg)
    {
        jelly_cube_derivative<GridWidth> result;

        for (size_t i = 0; i < node_count; ++i) {
            result.position_derivative[i] = arg.nodes[i].velocity;
            result.velocity_derivative[i] = simd_dvec3(0, gravity, 0);
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
    // TODO
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
}

// void foo()
// {
//     jelly_cube<15> cube;
//     cube.step(.3);
// }

} // end namespace akeley

#endif /* !AKELEY_JELLY_PHYSICS_HPP_ */
