// Adapts my new templatized jelly cube to the creaky old jellymjf.h interface.
// To be fair, this does more than just "adapt", since the old interface also
// required some OpenGL helper functions (make elements array, normals, etc.)
// and all this bead logic.
#include <algorithm>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include "jellymjf.h"
#include "physics.hpp"
#include "vec3.h"

using akeley::node;
using akeley::jelly_cube;

static inline Vec3 vec3(simd_dvec3 v)
{
    return vec3(X(v), Y(v), Z(v));
}

#ifndef GRID_WIDTH
#define GRID_WIDTH 15
#endif

#ifndef MAX_BEADS
#define MAX_BEADS 8192
#endif

constexpr size_t grid_width = GRID_WIDTH;

static jelly_cube<grid_width, 400, 1, 1> global_jelly_cube;
using grid_coordinate = decltype(global_jelly_cube)::grid_coordinate;

struct bead
{
    float red;
    float green;
    float blue;
    // Grid coordinates of the cell the bead is in.
    grid_coordinate cell_coordinate;
    // Normalized coordinate in the cell.
    simd_dvec3 blend;
};

static std::vector<bead> beads;
constexpr int max_beads = MAX_BEADS; // Required by old C-style interface.

int get_bead_count() { return beads.size(); }
int get_max_beads() { return max_beads; }
int get_max_lamps() { return 1; }

// I support just one of the mysterious lamps.
static simd_dvec3 lamp_position, lamp_normalized_direction;
static double lamp_heat_per_second, lamp_beam_radius;

// Add a bead with "original" world coordinate xyz and float color
// rgb.  The coordinate is "original" in the sense that this is where
// the bead would be in the original jello cube before it's
// deformed. If the cube is already deformed, the bead will appear
// wherever the point (x,y,z) in the original cube has deformed to.
//
// Returns:
//
// 0 on success
//
// ENOMEM if the bead array is full
//
// EDOM if the coordinates are not in the 1x1x1 cube.
int add_bead(float x, float y, float z, float r, float g, float b)
{
    simd_dvec3 input_coords = simd_dvec3(x, y, z);

    if (get_bead_count() >= max_beads) return ENOMEM;
    if (x < 0 || x > 1) return EDOM;
    if (y < 0 || y > 1) return EDOM;
    if (z < 0 || z > 1) return EDOM;
    if (!is_real(input_coords)) return EDOM;

    simd_dvec3 floaty_grid_coords = grid_width * input_coords;

    size_t cell_x = (size_t) X(floaty_grid_coords);
    size_t cell_y = (size_t) Y(floaty_grid_coords);
    size_t cell_z = (size_t) Z(floaty_grid_coords);

    double blend_x = fmod(X(floaty_grid_coords), 1.0);
    double blend_y = fmod(Y(floaty_grid_coords), 1.0);
    double blend_z = fmod(Z(floaty_grid_coords), 1.0);

    // Deal with float roundoff or case when input = 1.0.
    if (cell_x >= grid_width) {
        cell_x = grid_width - 1;
        blend_x = 1.0;
    }
    if (cell_y >= grid_width) {
        cell_y = grid_width - 1;
        blend_y = 1.0;
    }
    if (cell_z >= grid_width) {
        cell_z = grid_width - 1;
        blend_z = 1.0;
    }

    auto new_bead = bead {
        r, g, b,
        { cell_x, cell_y, cell_z },
        { blend_x, blend_y, blend_z } };

    beads.push_back(new_bead);
    return 0;
}

/*  Set lamp number [i] to have position [x, y, z], point in  the  direction
 *  of the vector [dx, dy, dz], and have the specified heating power.
 *      Returns:
 *  0 on success
 *  ENOMEM if the lamp number is out of range
 *  EDOM if the direction vector is 0, or any of the vectors are not real.
 */
int set_lamp(
    int i,
    float x, float y, float z,
    float dx, float dy, float dz,
    float heat_per_second,
    float beam_radius
) {
    // I only support 1 lamp.
    if (i != 0) return ENOMEM;

    double length;
    simd_dvec3 arg_position(x, y, z);
    simd_dvec3 arg_dv(dx, dy, dz);
    simd_dvec3 normalized_direction = normalize(arg_dv, &length);
    if (length <= 0.0 || !is_real(arg_dv) || !is_real(arg_position)) {
        return EDOM;
    }

    lamp_position = arg_position;
    lamp_normalized_direction = normalized_direction;

    lamp_heat_per_second = heat_per_second;
    lamp_beam_radius = beam_radius;

    return 0;
}

// Blend the 8 corner positions of the cell the bead is in to get its position.
static Vec3 bead_position(bead b)
{
    double x0wt = X(b.blend);
    double x1wt = 1 - x0wt;
    double y0wt = Y(b.blend);
    double y1wt = 1 - y0wt;
    double z0wt = Z(b.blend);
    double z1wt = 1 - z0wt;

    auto x = b.cell_coordinate.x;
    auto y = b.cell_coordinate.y;
    auto z = b.cell_coordinate.z;

    auto get_position_vector = [x, y, z] (
        unsigned x0_or_1,
        unsigned y0_or_1,
        unsigned z0_or_1)
    {
        return global_jelly_cube.get_node(
            { x + x0_or_1, y + y0_or_1, z + z0_or_1 } ).position;
    };

    auto v000 = get_position_vector(0, 0, 0);
    auto v001 = get_position_vector(0, 0, 1);
    auto v010 = get_position_vector(0, 1, 0);
    auto v011 = get_position_vector(0, 1, 1);
    auto v100 = get_position_vector(1, 0, 0);
    auto v101 = get_position_vector(1, 0, 1);
    auto v110 = get_position_vector(1, 1, 0);
    auto v111 = get_position_vector(1, 1, 1);

    auto vx00 = x0wt * v000 + x1wt * v100;
    auto vx01 = x0wt * v001 + x1wt * v101;
    auto vx10 = x0wt * v010 + x1wt * v110;
    auto vx11 = x0wt * v011 + x1wt * v111;

    auto vxy0 = y0wt * vx00 + y1wt * vx10;
    auto vxy1 = y0wt * vx01 + y1wt * vx11;

    auto vxyz = z0wt * vxy0 + z1wt * vxy1;
    return vec3(vxyz);
}

void reset_cube()
{
    global_jelly_cube.reset();
    beads.clear();
}

constexpr double jolt_ceiling = 0.1;

// Jolt the jello cube by adding the specified  velocity  vector  to  nodes
// touching the floor (actually, below the threshold jolt_ceiling).
void jolt(float dx, float dy, float dz)
{
    simd_dvec3 dv(dx, dy, dz);
    auto jolt_if_below_ceiling = [dv] (node* n, grid_coordinate)
    {
        double blend = Y(n->position) - jolt_ceiling;
        simd_dvec3 original_velocity = n->velocity;
        simd_dvec3 jolted_velocity = original_velocity + dv;
        n->velocity = step(jolted_velocity, original_velocity, blend);
    };

    global_jelly_cube.map_node_pointers(jolt_if_below_ceiling);
}

static float camera_center[3];

// Average the positions of the center 8 nodes of the global jelly cube.
static simd_dvec3 get_cube_center()
{
    auto& cube = global_jelly_cube;

    auto a = grid_width / 2;
    auto b = (grid_width + 1) / 2;
    
    auto position_at = [&cube] (grid_coordinate coord)
    {
        return cube.get_node(coord).position;
    };

    return 0.125 * (
        (( position_at( { a, a, a } ) + position_at( { a, a, b } ) )  +
         ( position_at( { a, b, a } ) + position_at( { a, b, b } ) )) +
        (( position_at( { b, a, a } ) + position_at( { b, a, b } ) ) +
         ( position_at( { b, b, a } ) + position_at( { b, b, b } ) )) );
}

// Update the global camera_center 3-vector with the average position
// the global cube's center was the last (up to) 600 times this was
// called.
static void update_camera_center()
{
    constexpr size_t max_sample_count = 600;
    static size_t sample_count = 0;
    static simd_dvec3 samples[max_sample_count];

    static size_t idx = 0;

    simd_dvec3 this_tick_center = get_cube_center();
    samples[idx++] = this_tick_center;
    sample_count = std::min(sample_count+1, max_sample_count);

    if (idx >= max_sample_count)
    {
        idx = 0;
    }
    
    simd_dvec3 total(0, 0, 0);
    for (size_t i = 0; i < sample_count; ++i) {
        total += samples[i];
    }
    simd_dvec3 average = total * (1.0 / sample_count);
    
    camera_center[0] = float(X(average));
    camera_center[1] = float(Y(average));
    camera_center[2] = float(Z(average));
}

// Tick the simulation by a time step dt.
// Also updates the recommended center position of the camera
// (average position the center node had in the last few ticks).
void tick(float dt)
{
    global_jelly_cube.step_by_dt(dt);
    // TODO: Apply heat.
    update_camera_center();
}

float const* get_camera_center()
{
    return camera_center;
}

// Code ahead handles exporting data from the simulation to the client that
// draws  the  cube and beads. Functions update and return pointers to data
// in a format suitable for being directly passed to glBufferData.
// GLOBAL ARRAYS EVERYWHERE.

// 3 vertices per triangle, 6 faces, 2*GRID_WIDTH*GRID_WIDTH triangles per face
#define FACE_ELEMENT_COUNT  (3 * 2 * GRID_WIDTH * GRID_WIDTH)
#define CUBE_ELEMENT_COUNT  (6 * FACE_ELEMENT_COUNT)
// 6 faces, (GRID_WIDTH+1)*(GRID_WIDTH*1) vertices per face.
// Edges and corners don't share vertices due to differing normals.
#define FACE_VERTEX_COUNT  ((GRID_WIDTH+1) * (GRID_WIDTH+1))
#define CUBE_VERTEX_COUNT  (6 * FACE_VERTEX_COUNT)

static uint16_t cube_element_array[CUBE_ELEMENT_COUNT];
static CubeVertex cube_vertex_array[CUBE_VERTEX_COUNT];
static BeadVertex bead_vertex_array[MAX_BEADS];

// Return data needed to pass vertex/element arrays to OpenGL VBO.
int get_cube_element_count(void) { return CUBE_ELEMENT_COUNT; }
int get_cube_vertex_count(void) { return CUBE_VERTEX_COUNT; }

// We are using interleaved arrays. Position data will always be at offset 0.
// These functions tell us the byte offset of the other vertex data, and the
// overall stride between one vertex and the next.
int get_cube_vertex_stride(void) { return sizeof(CubeVertex); }
int get_cube_normal_offset(void) {
    return (char*)(&cube_vertex_array[0].normal)
         - (char*)(&cube_vertex_array[0]);
}
int get_bead_vertex_stride(void) { return sizeof(BeadVertex); }
int get_bead_color_offset(void) {
    return (char*)(&bead_vertex_array[0].color)
         - (char*)(&bead_vertex_array[0]);
}


/*  Return a pointer to a  statically-allocated  array  of  16-bit  elements
 *  suitable  for use in glDrawElements in conjunction with vertex data from
 *  the next function (update_cube_vertices). The node positions  from  that
 *  function  are  connected into little triangles in the logical way; these
 *  triangles always have CCW winding when viewed from outside of the  cube.
 *  This function always returns the same data.
 */
uint16_t const* get_cube_elements(void) {
    // The six partitions of the element array corresponding to 6 cube faces.
    uint16_t* x0_face = &cube_element_array[0 * FACE_ELEMENT_COUNT];
    uint16_t* x1_face = &cube_element_array[1 * FACE_ELEMENT_COUNT];
    uint16_t* y0_face = &cube_element_array[2 * FACE_ELEMENT_COUNT];
    uint16_t* y1_face = &cube_element_array[3 * FACE_ELEMENT_COUNT];
    uint16_t* z0_face = &cube_element_array[4 * FACE_ELEMENT_COUNT];
    uint16_t* z1_face = &cube_element_array[5 * FACE_ELEMENT_COUNT];
    
    int index = 0;
    
    // Fill in the element array six squares at a time.
    // 2*3 = 6 vertices per square, and we fill the 6 faces' squares
    // in parallel.
    for (int b = 0; b < GRID_WIDTH; ++b) {
        for (int a = 0; a < GRID_WIDTH; ++a) {
            z1_face[index] = FACE_VERTEX_COUNT +
            (z0_face[index] = FACE_VERTEX_COUNT +
            (y1_face[index] = FACE_VERTEX_COUNT +
            (y0_face[index] = FACE_VERTEX_COUNT +
            (x1_face[index] = FACE_VERTEX_COUNT +
            (x0_face[index] = a + b * (GRID_WIDTH+1))))));
            ++index;
            
            z1_face[index] = FACE_VERTEX_COUNT +
            (z0_face[index] = FACE_VERTEX_COUNT +
            (y1_face[index] = FACE_VERTEX_COUNT +
            (y0_face[index] = FACE_VERTEX_COUNT +
            (x1_face[index] = FACE_VERTEX_COUNT +
            (x0_face[index] = (a+1) + (b+1) * (GRID_WIDTH+1))))));
            ++index;
            
            z1_face[index] = FACE_VERTEX_COUNT +
            (z0_face[index] = FACE_VERTEX_COUNT +
            (y1_face[index] = FACE_VERTEX_COUNT +
            (y0_face[index] = FACE_VERTEX_COUNT +
            (x1_face[index] = FACE_VERTEX_COUNT +
            (x0_face[index] = a + (b+1) * (GRID_WIDTH+1))))));
            ++index;
            
            z1_face[index] = FACE_VERTEX_COUNT +
            (z0_face[index] = FACE_VERTEX_COUNT +
            (y1_face[index] = FACE_VERTEX_COUNT +
            (y0_face[index] = FACE_VERTEX_COUNT +
            (x1_face[index] = FACE_VERTEX_COUNT +
            (x0_face[index] = a + b * (GRID_WIDTH+1))))));
            ++index;
            
            z1_face[index] = FACE_VERTEX_COUNT +
            (z0_face[index] = FACE_VERTEX_COUNT +
            (y1_face[index] = FACE_VERTEX_COUNT +
            (y0_face[index] = FACE_VERTEX_COUNT +
            (x1_face[index] = FACE_VERTEX_COUNT +
            (x0_face[index] = (a+1) + b * (GRID_WIDTH+1))))));
            ++index;
            
            z1_face[index] = FACE_VERTEX_COUNT +
            (z0_face[index] = FACE_VERTEX_COUNT +
            (y1_face[index] = FACE_VERTEX_COUNT +
            (y0_face[index] = FACE_VERTEX_COUNT +
            (x1_face[index] = FACE_VERTEX_COUNT +
            (x0_face[index] = (a+1) + (b+1) * (GRID_WIDTH+1))))));
            ++index;
        }
    }
    
    return cube_element_array;
}

// Calculate displacement vector [to] - [from].
static inline Vec3 node_sub(node const* to, node const* from) {
    return vec3(to->position - from->position);
}

/*  Recalculate the positions and normals of nodes at  the  surface  of  the
 *  cube. Always returns the same pointer to a statically-allocated array of
 *  vertex data. All triangles have CCW winding when viewed from outside the
 *  cube.
 */
// 2019: At this point it's just "if it ain't broke, don't fix it". I
// just shoehorned in the changes needed to make it work with the new
// jelly cube.
CubeVertex const* update_cube_vertices(void) {
    // Shoehorned-in thing -- looks up node reference from node coordinate.
    auto node_ref = [] (size_t x, size_t y, size_t z) -> const node&
    {
        return global_jelly_cube.get_node( { x, y, z } );
    };
    
    // The six partitions of the vertex array corresponding to 6 cube faces.
    CubeVertex* x0_face = &cube_vertex_array[0 * FACE_VERTEX_COUNT];
    CubeVertex* x1_face = &cube_vertex_array[1 * FACE_VERTEX_COUNT];
    CubeVertex* y0_face = &cube_vertex_array[2 * FACE_VERTEX_COUNT];
    CubeVertex* y1_face = &cube_vertex_array[3 * FACE_VERTEX_COUNT];
    CubeVertex* z0_face = &cube_vertex_array[4 * FACE_VERTEX_COUNT];
    CubeVertex* z1_face = &cube_vertex_array[5 * FACE_VERTEX_COUNT];
    
    // First fill in the positions of the vertices, and zero
    // out the normal vector in preparation for calculating normals.
    // Note that we're working per-vertex, not per triangle, so
    // the coordinate range is 0...GRID_WIDTH inclusive, not exclusive.
    // As before we fill the 6 face partitions in parallel.
    for (size_t b = 0; b <= GRID_WIDTH; ++b) {
        for (size_t a = 0; a <= GRID_WIDTH; ++a) {
            size_t index = a + b*(GRID_WIDTH+1);
            
            x0_face[index].position = vec3(node_ref(0, b, a).position);
            x0_face[index].normal = vec3(0,0,0);
            
            x1_face[index].position = vec3(node_ref(GRID_WIDTH, a, b).position);
            x1_face[index].normal = vec3(0,0,0);
            
            y0_face[index].position = vec3(node_ref(a, 0, b).position);
            y0_face[index].normal = vec3(0,0,0);
            
            y1_face[index].position = vec3(node_ref(b, GRID_WIDTH, a).position);
            y1_face[index].normal = vec3(0,0,0);
            
            z0_face[index].position = vec3(node_ref(b, a, 0).position);
            z0_face[index].normal = vec3(0,0,0);
            
            z1_face[index].position = vec3(node_ref(a, b, GRID_WIDTH).position);
            z1_face[index].normal = vec3(0,0,0);
        }
    }
    
    // For each triangular face, add its normal to its 3 corners. This results
    // in inner vertices having a normal that is a blend of its neighbors,
    // though some triangles will have more influence than others because
    // I'm too cheap to normalize the normal vectors.
    for (int b = 0; b < GRID_WIDTH; ++b) {
        for (int a = 0; a < GRID_WIDTH; ++a) {
            // We're looking at the square with grid coordinates
            // (a, b) to (a+1, b+1). i00 i01 ... i11 are the indicies
            // within the face arrays corresponding to the nodes
            // with grid coordinates (a, b), (a, b+1) ... (a+1, b+1).
            // v01, v11, v10 are the differences in position between the
            // nodes with grid coordinates (a, b+1), (a+1, b+1), (a+1, b)
            // and the node with grid coordinate (a, b).
            // (These 2D grid coordinates are the 3D coordinates with the
            // constant coordinate for a face removed, e.g., (a,b) is grid
            // coordinate (a, 0, b) for y=0 face). Note that sometimes a/b
            // are reversed in order to keep the normals and front
            // (anticlockwise) face pointing the right way.
            int i00 = a     + b * (GRID_WIDTH+1);
            int i01 = a     + (b+1) * (GRID_WIDTH+1);
            int i10 = a + 1 + b * (GRID_WIDTH+1);
            int i11 = a + 1 + (b+1) * (GRID_WIDTH+1);
            
            Vec3 v01, v11, v10, normal0, normal1;
            const node* n00;
            
#define ADD_TRIANGLE_NORMALS(face) \
            normal0 = cross(v11, v01); \
            normal1 = cross(v10, v11); \
            iadd(&face[i00].normal, normal0); \
            iadd(&face[i11].normal, normal0); \
            iadd(&face[i01].normal, normal0); \
            iadd(&face[i00].normal, normal1); \
            iadd(&face[i10].normal, normal1); \
            iadd(&face[i11].normal, normal1)
            
            n00 = &node_ref(0, b, a);
            v01 = node_sub(&node_ref(0, b+1, a), n00);
            v11 = node_sub(&node_ref(0, b+1, a+1), n00);
            v10 = node_sub(&node_ref(0, b, a+1), n00);
            ADD_TRIANGLE_NORMALS(x0_face);
            
            n00 = &node_ref(GRID_WIDTH, a, b);
            v01 = node_sub(&node_ref(GRID_WIDTH, a, b+1), n00);
            v11 = node_sub(&node_ref(GRID_WIDTH, a+1, b+1), n00);
            v10 = node_sub(&node_ref(GRID_WIDTH, a+1, b), n00);
            ADD_TRIANGLE_NORMALS(x1_face);
            
            n00 = &node_ref(a, 0, b);
            v01 = node_sub(&node_ref(a, 0, b+1), n00);
            v11 = node_sub(&node_ref(a+1, 0, b+1), n00);
            v10 = node_sub(&node_ref(a+1, 0, b), n00);
            ADD_TRIANGLE_NORMALS(y0_face);
            
            n00 = &node_ref(b, GRID_WIDTH, a);
            v01 = node_sub(&node_ref(b+1, GRID_WIDTH, a), n00);
            v11 = node_sub(&node_ref(b+1, GRID_WIDTH, a+1), n00);
            v10 = node_sub(&node_ref(b, GRID_WIDTH, a+1), n00);
            ADD_TRIANGLE_NORMALS(y1_face);
            
            n00 = &node_ref(a, b, 0);
            v01 = node_sub(&node_ref(b+1, a, 0), n00);
            v11 = node_sub(&node_ref(b+1, a+1, 0), n00);
            v10 = node_sub(&node_ref(b, a+1, 0), n00);
            ADD_TRIANGLE_NORMALS(z0_face);
            
            n00 = &node_ref(a, b, GRID_WIDTH);
            v01 = node_sub(&node_ref(a, b+1, GRID_WIDTH), n00);
            v11 = node_sub(&node_ref(a+1, b+1, GRID_WIDTH), n00);
            v10 = node_sub(&node_ref(a+1, b, GRID_WIDTH), n00);
            ADD_TRIANGLE_NORMALS(z1_face);
        }
    }
    
    return cube_vertex_array;
}

/*  Refill the bead vertex array with the latest data.  Always  returns  the
 *  same pointer to a static array of bead vertices.
 */
BeadVertex const* update_bead_vertices(void) {
    size_t i = 0;
    for (const bead& b : beads) {
        if (i >= MAX_BEADS) break;
        bead_vertex_array[i].position = bead_position(b);
        bead_vertex_array[i].color = vec3(b.red, b.green, b.blue);
    }
    return bead_vertex_array;
}

/*  Alternate bead arrays that are filled with 1 bead per  node,  with  that
 *  bead  colored  based  on the node's temperature. There's an additional 4
 *  beads to help indicate the world coordinate axes.
 */
#define DEBUG_BEAD_COUNT  ((GRID_WIDTH+1)*(GRID_WIDTH+1)*(GRID_WIDTH+1) + 4)
int get_debug_bead_count(void) {
    return DEBUG_BEAD_COUNT;
}
BeadVertex const* update_debug_bead_vertices(void) {
    static BeadVertex array[DEBUG_BEAD_COUNT];
    array[0].position = vec3(0, 0, 0);
    array[0].color = vec3(1, 1, 1);
    array[1].position = vec3(1.2f, 0, 0);
    array[1].color = vec3(1, 0, 0);
    array[2].position = vec3(0, 1.2f, 0);
    array[2].color = vec3(0, 1, 0);
    array[3].position = vec3(0, 0, 1.2f);
    array[3].color = vec3(0.3f, 0.6f, 1);
    size_t index = 4;
    
    auto append_debug_bead = [&index] (const node& n, grid_coordinate)
    {
        using akeley::ambient_temperature;
        
        assert(index < DEBUG_BEAD_COUNT);
        BeadVertex& vert = array[index++];
        
        vert.position = vec3(n.position);
        float temperature = float(n.temperature);
        
        float r = (temperature - ambient_temperature)*.01f;
        float g = 0.4f;
        float b = 1.0f - (temperature - ambient_temperature)*.01f;
        
        if (r < 0.0f) r = 0.0f;
        if (r > 1.0f) r = 1.0f;
        if (b < 0.0f) b = 0.0f;
        if (b > 1.0f) b = 1.0f;
        
        vert.color = vec3(r, g, b);
    };
    
    global_jelly_cube.map_nodes(append_debug_bead);
    return array;
}

void hack_zero_velocities()
{
    global_jelly_cube.hack_zero_velocities();
}
