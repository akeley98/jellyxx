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
constexpr double ambient_temperature = 20.0;
constexpr double node_mass = 1.0;
constexpr double non_floor_dampening = 11;
constexpr double floor_dampening = 11;

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
struct grid_coordinate_t
{
    constexpr static auto narrowest_unsigned_impl()
    {
        if constexpr(GridWidth <= 0xFF) {
            return uint8_t(0);
        } else if constexpr (GridWidth <= 0xFFFF) {
            return uint16_t(0);
        } else if constexpr (GridWidth <= 0xFFFF'FFFF) {
            return uint32_t(0);
        } else if constexpr (GridWidth <= 0xFFFF'FFFF'FFFF'FFFF) {
            return uint64_t(0);
        } else {
            typedef unsigned long long ull;
            return ull(0);
        }
    }

    using value_type = decltype(narrowest_unsigned_impl());

    value_type x;
    value_type y;
    value_type z;

    grid_coordinate_t(size_t x_, size_t y_, size_t z_) :
        x(value_type(x_)),
        y(value_type(y_)),
        z(value_type(z_))
    {
        assert(x_ <= GridWidth);
        assert(y_ <= GridWidth);
        assert(z_ <= GridWidth);
    }

    size_t to_index() const
    {
        size_t gw1 = GridWidth + 1;
        return x*(gw1*gw1) + y*gw1 + z;
    }
};

template <
    size_t GridWidth,
    size_t BaseK,
    size_t NodeMassNumerator,
    size_t NodeMassDenominator>
class jelly_cube
{
    static constexpr size_t node_count =
        (GridWidth+1) * (GridWidth+1) * (GridWidth+1);
    static constexpr double node_mass =
        NodeMassNumerator / double(NodeMassDenominator);
    static constexpr double node_mass_recip =
        NodeMassDenominator / double(NodeMassNumerator);

    std::array<node, node_count> nodes;
    std::shared_ptr<const std::vector<spring>> springs_ptr;
    // Note this vector is always the same for all jelly_cubes of a given
    // GridWidth; can be replaced with a constexpr std::vector in C++20.

  public:
    struct derivative
    {
        std::array<simd_dvec3, node_count> position_derivative;
        std::array<simd_dvec3, node_count> velocity_derivative;

        derivative operator* (double scale)
        {
            derivative result;
            for (size_t i = 0; i < node_count; ++i) {
                result.position_derivative[i] = scale * position_derivative[i];
                result.velocity_derivative[i] = scale * velocity_derivative[i];
            }
            return result;
        }
    };

    using grid_coordinate = grid_coordinate_t<GridWidth>;

    jelly_cube()
    {
        springs_ptr =
            std::make_shared<std::vector<spring>>(make_springs());
        reset();
    }

    void hack_zero_velocities()
    {
        auto set_to_zero_velocity = [] (node* n, grid_coordinate)
        { n->velocity = simd_dvec3(0, 0, 0); };

        map_node_pointers(set_to_zero_velocity);
    }

    void reset()
    {
        double double_grid_width = GridWidth;
        auto init_node =
        [double_grid_width] (node* n, grid_coordinate coord)
        {
            node new_node;
            new_node.position = simd_dvec3(
                coord.x / double_grid_width,
                coord.y / double_grid_width,
                coord.z / double_grid_width);
            *n = new_node;
        };
        map_node_pointers(init_node);
    }

    // Get rid of the move constructor to avoid null shared_ptr
    // (there's not much performance benefit of moves anyway for
    // std::array, the other member variable).
    jelly_cube(const jelly_cube&) = default;
    jelly_cube& operator= (const jelly_cube&) = default;
    ~jelly_cube() = default;

    const node& get_node(grid_coordinate coord) const
    {
        return nodes[coord.to_index()];
    }

    node* view_node(grid_coordinate coord) {
        return const_cast<node*>(&get_node(coord));
    }

    void step_by_dt(double dt)
    {
        *this = rk4_step(*this, calculate_derivative, dt);
        enforce_floor();

        auto vel = get_node( { GridWidth, GridWidth, GridWidth } ).velocity;
        printf("%f %f %f\n", X(vel), Y(vel), Z(vel));
    }

    jelly_cube operator+ (const derivative& deriv) const
    {
        jelly_cube result = *this;
        return result += deriv;
    }

    jelly_cube&
    operator+= (const derivative& deriv) noexcept
    {
        for (size_t i = 0; i < node_count; ++i) {
            nodes[i].position += deriv.position_derivative[i];
            nodes[i].velocity += deriv.velocity_derivative[i];
        }
        return *this;
    }

    template <typename CallableWithNodePointerAndGridCoordinate>
    void map_node_pointers(CallableWithNodePointerAndGridCoordinate& f)
    {
        size_t idx = 0;
        for (size_t x = 0; x <= GridWidth; ++x) {
            for (size_t y = 0; y <= GridWidth; ++y) {
                for (size_t z = 0; z <= GridWidth; ++z, ++idx) {
                    grid_coordinate coord(x, y, z);
                    assert(coord.to_index() == idx);
                    f(&nodes[idx], coord);
                }
            }
        }
    }

    template <typename CallableWithNodeAndGridCoordinate>
    void map_nodes(CallableWithNodeAndGridCoordinate& f) const
    {
        size_t idx = 0;
        for (size_t x = 0; x <= GridWidth; ++x) {
            for (size_t y = 0; y <= GridWidth; ++y) {
                for (size_t z = 0; z <= GridWidth; ++z, ++idx) {
                    grid_coordinate coord(x, y, z);
                    assert(coord.to_index() == idx);
                    f(nodes[idx], coord);
                }
            }
        }
    }

  private:
    static std::vector<spring> make_springs()
    {
        constexpr double edge_spring_k = BaseK;
        constexpr double face_spring_k = BaseK * 1.4142135623730951;
        constexpr double inner_spring_k = BaseK * 1.7320508075688772;

        std::vector<spring> springs;
        auto to_index = [] (size_t x, size_t y, size_t z)
        {
            return grid_coordinate(x, y, z).to_index();
        };

        spring_spec horizontal_edge_spring { edge_spring_k, 1.0 / GridWidth },
            face_spring { face_spring_k, 1.4142135623730951 / GridWidth },
            inner_spring { inner_spring_k, 1.7320508075688772 / GridWidth };

        auto maybe_push = [&springs, to_index] (
            size_t x0,
            size_t y0,
            size_t z0,
            size_t x1,
            size_t y1,
            size_t z1,
            spring_spec spec)
        {
            if (x0 > GridWidth || y0 > GridWidth || z0 > GridWidth) {
                return;
            }
            if (x1 > GridWidth || y1 > GridWidth || z1 > GridWidth) {
                return;
            }
            spring spr { to_index(x0, y0, z0), to_index(x1, y1, z1) , spec };
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

        for (size_t x = 0; x <= GridWidth; ++x) {
            for (size_t y = 0; y <= GridWidth; ++y) {
                for (size_t z = 0; z <= GridWidth; ++z) {
                    maybe_push(x, y, z, x, y, z+1, horizontal_edge_spring);
                    maybe_push(x, y, z, x, y+1, z, vertical_edge_spring(y));
                    maybe_push(x, y, z, x+1, y, z, horizontal_edge_spring);

                    maybe_push(x, y, z, x+1, y+1, z, face_spring);
                    maybe_push(x+1, y, z, x, y+1, z, face_spring);
                    maybe_push(x, y, z, x+1, y, z+1, face_spring);
                    maybe_push(x+1, y, z, x, y, z+1, face_spring);
                    maybe_push(x, y, z, x, y+1, z+1, face_spring);
                    maybe_push(x, y+1, z, x, y, z+1, face_spring);

                    maybe_push(x, y, z, x+1, y+1, z+1, inner_spring);
                    maybe_push(x+1, y, z, x, y+1, z+1, inner_spring);
                    maybe_push(x, y+1, z, x+1, y, z+1, inner_spring);
                    maybe_push(x, y, z+1, x+1, y+1, z, inner_spring);
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

    static derivative
    calculate_derivative(const jelly_cube& arg)
    {
        derivative result;

        for (size_t i = 0; i < node_count; ++i) {
            result.position_derivative[i] = arg.nodes[i].velocity;
            result.velocity_derivative[i] = simd_dvec3(0, -gravity, 0);
        }

        for (const spring& the_spring : *arg.springs_ptr) {
            arg.add_spring_force_derivative(&result, the_spring);
        }

        for (size_t x = 0; x < GridWidth; ++x) {
            for (size_t y = 0; y < GridWidth; ++y) {
                for (size_t z = 0; z < GridWidth; ++z) {
                    arg.add_cell_pressure_derivative(&result, {x, y, z});
                }
            }
        }

        arg.add_friction_dampening_derivative(&result);

        return result;
    }

    void add_spring_force_derivative(
        derivative* deriv,
        spring the_spring) const
    {
        const node& node_0 = nodes.at(the_spring.node_index_0);
        const node& node_1 = nodes.at(the_spring.node_index_1);

        double distance;
        simd_dvec3 displacement = node_1.position - node_0.position;
        simd_dvec3 unit_displacement = normalize(displacement, &distance);
        auto stretch_distance = distance - the_spring.spec.length;
        
        auto fudge_distance = 0.04 / GridWidth;
        if (stretch_distance > 0) {
            stretch_distance = std::max(0.0, stretch_distance - fudge_distance);
        } else {
            stretch_distance = std::min(0.0, stretch_distance + fudge_distance);
        }
        
        auto kX = the_spring.spec.k * stretch_distance;
        auto force = kX * node_0.strength * node_1.strength;
        auto accel = force * node_mass_recip;
        simd_dvec3 added_vector = accel * unit_displacement;

        deriv->velocity_derivative[the_spring.node_index_0] += added_vector;
        deriv->velocity_derivative[the_spring.node_index_1] -= added_vector;
    }

    /*  Calculate a cell's pressure given its volume */
    static inline float pressure(double volume) {
        double stretch = volume * (GridWidth * GridWidth * GridWidth);
        double stretch_fudge = 0.06;
        
        if (stretch > 1) {
            stretch = std::max(1.0, stretch - stretch_fudge);
        } else {
            stretch = std::min(1.0, stretch + stretch_fudge);
        }
        
        return clamp_float(1 - stretch, -1, +1) * 80000.0f;
    }

    void add_cell_pressure_derivative(
        derivative* deriv,
        grid_coordinate coord) const
    {
        auto x = coord.x;
        auto y = coord.y;
        auto z = coord.z;

        simd_dvec3 p000 = get_node( { x, y, z } ).position;
        simd_dvec3 p001 = get_node( { x, y, z+1u } ).position;
        simd_dvec3 p010 = get_node( { x, y+1u, z } ).position;
        simd_dvec3 p011 = get_node( { x, y+1u, z+1u } ).position;
        simd_dvec3 p100 = get_node( { x+1u, y, z } ).position;
        simd_dvec3 p101 = get_node( { x+1u, y, z+1u } ).position;
        simd_dvec3 p110 = get_node( { x+1u, y+1u, z } ).position;
        simd_dvec3 p111 = get_node( { x+1u, y+1u, z+1u } ).position;

        auto ref_dv_vector = [deriv] (size_t x, size_t y, size_t z)
        -> simd_dvec3&
        {
            size_t index = grid_coordinate(x, y, z).to_index();
            return deriv->velocity_derivative[index];
        };

        // velocity derivative vectors (to modify) corresponding to above nodes.
        simd_dvec3& accel000 = ref_dv_vector(x, y, z);
        simd_dvec3& accel001 = ref_dv_vector(x, y, z+1);
        simd_dvec3& accel010 = ref_dv_vector(x, y+1, z);
        simd_dvec3& accel011 = ref_dv_vector(x, y+1, z+1);
        simd_dvec3& accel100 = ref_dv_vector(x+1, y, z);
        simd_dvec3& accel101 = ref_dv_vector(x+1, y, z+1);
        simd_dvec3& accel110 = ref_dv_vector(x+1, y+1, z);
        simd_dvec3& accel111 = ref_dv_vector(x+1, y+1, z+1);

        // Vectors from node[x][y][z] to the other 7 nodes.
        simd_dvec3 v000_001 = p001 - p000;
        simd_dvec3 v000_011 = p011 - p000;
        simd_dvec3 v000_010 = p010 - p000;
        simd_dvec3 v000_110 = p110 - p000;
        simd_dvec3 v000_100 = p100 - p000;
        simd_dvec3 v000_101 = p101 - p000;
        simd_dvec3 v000_111 = p111 - p000;

        // Paralellogram vectors for 6 triangles around node 000.
        simd_dvec3 pvec_000_001_011 = cross(v000_001, v000_011);
        simd_dvec3 pvec_000_011_010 = cross(v000_011, v000_010);
        simd_dvec3 pvec_000_010_110 = cross(v000_010, v000_110);
        simd_dvec3 pvec_000_110_100 = cross(v000_110, v000_100);
        simd_dvec3 pvec_000_100_101 = cross(v000_100, v000_101);
        simd_dvec3 pvec_000_101_001 = cross(v000_101, v000_001);

        // Sum of the dot products using less-than-obvious algorithm.
        auto s =   elementwise_mul(v000_111, pvec_000_001_011);
        auto t =   elementwise_mul(v000_111, pvec_000_011_010);
        s = s + elementwise_mul(v000_111, pvec_000_010_110);
        t = t + elementwise_mul(v000_111, pvec_000_110_100);
        s = s + elementwise_mul(v000_111, pvec_000_100_101);
        t = t + elementwise_mul(v000_111, pvec_000_101_001);
        simd_dvec3 SS = s + t;

        const float volume = (-1.0/6.0) * (X(SS) + Y(SS) + Z(SS));
        const float quarter_pressure = 0.25f * pressure(volume);

        // Add forces onto the 3 nodes of each triangle. The right-angle node
        // (across from hypotenuse) gets 1/2 of the force and gets added first.
        // The other two nodes come next and get 1/4 each.
        simd_dvec3 accel0, accel1;

        accel001 += (accel0 = quarter_pressure * node_mass_recip * pvec_000_001_011);
        accel011 += (accel0 = 0.5 * accel0);
        accel000 += accel0;

        accel010 += (accel1 = quarter_pressure * node_mass_recip * pvec_000_011_010);
        accel011 += (accel1 = 0.5 * accel1);
        accel000 += accel1;

        accel010 += (accel0 = quarter_pressure * node_mass_recip * pvec_000_010_110);
        accel110 += (accel0 = 0.5 * accel0);
        accel000 += accel0;

        accel100 += (accel1 = quarter_pressure * node_mass_recip * pvec_000_110_100);
        accel110 += (accel1 = 0.5 * accel1);
        accel000 += accel1;

        accel100 += (accel0 = quarter_pressure * node_mass_recip * pvec_000_100_101);
        accel101 += (accel0 = 0.5 * accel0);
        accel000 += accel0;

        accel001 += (accel1 = quarter_pressure * node_mass_recip * pvec_000_101_001);
        accel101 += (accel1 = 0.5 * accel1);
        accel000 += accel1;

        // Now add pressure force for the other 6 triangles. I hope that
        // the compiler(s) used have a good register allocation strategy.
        simd_dvec3 v111_001 = p001 - p111;
        simd_dvec3 v111_011 = p011 - p111;
        simd_dvec3 v111_010 = p010 - p111;
        simd_dvec3 v111_110 = p110 - p111;
        simd_dvec3 v111_100 = p100 - p111;
        simd_dvec3 v111_101 = p101 - p111;

        simd_dvec3 pvec_111_001_101 = cross(v111_001, v111_101);
        simd_dvec3 pvec_111_101_100 = cross(v111_101, v111_100);
        simd_dvec3 pvec_111_100_110 = cross(v111_100, v111_110);
        simd_dvec3 pvec_111_110_010 = cross(v111_110, v111_010);
        simd_dvec3 pvec_111_010_011 = cross(v111_010, v111_011);
        simd_dvec3 pvec_111_011_001 = cross(v111_011, v111_001);

        accel101 += (accel0 = quarter_pressure * node_mass_recip * pvec_111_001_101);
        accel001 += (accel0 = 0.5 * accel0);
        accel111 += accel0;

        accel101 += (accel1 = quarter_pressure * node_mass_recip * pvec_111_101_100);
        accel100 += (accel1 = 0.5 * accel1);
        accel111 += accel1;

        accel110 += (accel0 = quarter_pressure * node_mass_recip * pvec_111_100_110);
        accel100 += (accel0 = 0.5 * accel0);
        accel111 += accel0;

        accel110 += (accel1 = quarter_pressure * node_mass_recip * pvec_111_110_010);
        accel010 += (accel1 = 0.5 * accel1);
        accel111 += accel1;

        accel011 += (accel0 = quarter_pressure * node_mass_recip * pvec_111_010_011);
        accel010 += (accel0 = 0.5 * accel0);
        accel111 += accel0;

        accel011 += (accel1 = quarter_pressure * node_mass_recip * pvec_111_011_001);
        accel001 += (accel1 = 0.5 * accel1);
        accel111 += accel1;
    }

    void add_friction_dampening_derivative(derivative* deriv) const
    {
        auto add_dampening = [deriv] (const node& n, grid_coordinate coord)
        {
            double dampening = Y(n.position) <= 0
                ? floor_dampening
                : non_floor_dampening;

            simd_dvec3 dv = -dampening * n.velocity;
            deriv->velocity_derivative[coord.to_index()] += dv;
        };

        map_nodes(add_dampening);
    }
};

} // end namespace akeley

#endif /* !AKELEY_JELLY_PHYSICS_HPP_ */
