#include <cuda/std/cmath>
#include <ctime>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <curand_kernel.h>

#define CUDA_CALL(x)                                                                                                    \
    do                                                                                                                  \
    {                                                                                                                   \
        if (x != cudaSuccess)                                                                                           \
        {                                                                                                               \
            std::cout << "CUDA error: " << cudaGetErrorString(x) << " in file " << __FILE__ << " at line " << __LINE__; \
            exit(x);                                                                                                    \
        }                                                                                                               \
    } while (0);

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GAMMA 2.2

#define CLIP_NEAR 1e-4
#define CLIP_FAR 1e20

#define MAX_BOUNCES 8
#define RANDOMNESS_ACCURACY 10
#define MIN_BRIGHTNESS 0.005

#define WIDTH 1000
#define HEIGHT 1000
#define SAMPLES 1500
#define FOV 55
#define DOF 0.01
#define FD 3.3 
#define TILE_WIDTH_X 32
#define TILE_WIDTH_Y 16
#define TILE_X (int)ceil(WIDTH / (double)TILE_WIDTH_X)
#define TILE_Y (int)ceil(HEIGHT / (double)TILE_WIDTH_Y)

#define MIX_SHADERS(a, b, fac, randState) ((curand_uniform(&randState) < fac) ? b : a)

struct Vec
{
    double x, y, z;

    __host__ __device__ Vec() : x(0), y(0), z(0) {}
    __host__ __device__ Vec(double x, double y, double z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
    __host__ __device__ Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
    __host__ __device__ Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }
    __host__ __device__ Vec norm() { return Vec(x, y, z) * (1 / cuda::std::sqrt(x * x + y * y + z * z)); }
    __host__ __device__ double dot(Vec b) const { return x * b.x + y * b.y + z * b.z; }
    __host__ __device__ Vec cross(Vec b) { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
};

struct Col
{
    double r, g, b;

    __host__ __device__ Col() : r(0), g(0), b(0) {}
    __host__ __device__ Col(double r, double g, double b) : r(r), g(g), b(b) {}

    __host__ __device__ Col operator+(const Col &y) const { return Col(r + y.r, g + y.g, b + y.b); }
    __host__ __device__ Col operator-(const Col &y) const { return Col(r - y.r, g - y.g, b - y.b); }
    __host__ __device__ Col operator*(const Col &y) const { return Col(r * y.r, g * y.g, b * y.b); }
    __host__ __device__ Col operator*(double y) const { return Col(r * y, g * y, b * y); }
};

struct Material
{
    Col diff, em;
    double metal, rough, ior, tr;

    __host__ __device__ Material() {}
    __host__ __device__ Material(Col diff, Col em, double metal, double rough, double ior, double tr) : diff(diff), em(em), metal(metal), rough(rough), ior(ior), tr(tr) {}
};

struct Ray
{
    Vec o, d;
    
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(Vec o, Vec d) : o(o), d(d) {}
};

struct ShaderOutput {
    Vec d;
    Col F;
    __device__ ShaderOutput() {}
    __device__ ShaderOutput(Vec d, Col F) : d(d), F(F) {}
};

struct Sphere
{
    double rad;
    Vec pos;
    Material material;

    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(double rad, Vec pos, Material material) : rad(rad), pos(pos), material(material) {}

    __device__ double intersect(Ray r)
    {
        //solving quadratic
        Vec op = pos - r.o;
        double t;
        double b = op.dot(r.d);                      // b/2 in the formula
        double dis = b * b - op.dot(op) + rad * rad; //(b^2 - 4ac)/4
        if (dis < 0)
            return 0;
        dis = cuda::std::sqrt(dis);
        return (t = b - dis) > CLIP_NEAR ? t : ((t = b + dis) > CLIP_NEAR ? t : 0);
    }
};



__device__ double clamp(double x) { return (x < 1) ? ((x > 0) ? x : 0) : 1; }
__device__ double toInt(double x) { return cuda::std::pow(clamp(x), 1 / GAMMA) * 255 + .5; }

__device__ bool intersect(Sphere spheres[], int numSpheres, Ray r, double &t, int &id) {
    double d; //intersecting all spheres
    t = CLIP_FAR;
    for (int i = 0; i < numSpheres; i++) {
        d = spheres[i].intersect(r);
        if (d != 0 && d < t) {
            t = d;
            id = i;
        }
    }
    return (t < CLIP_FAR);
}

__device__ Col mix(Col a, Col b, double fac) {
    return a + (b - a) * fac; //linear interpolation
}

__device__ double mix(double a, double b, double fac) {
    return a + (b - a) * fac; //linear interpolation
}

__device__ double fresnel(double ior, Vec rayDir, Vec normal) {
    normal = normal.dot(rayDir) > 0 ? normal * -1 : normal;
    double R = (1 - ior) / (1 + ior);
    R *= R;
    double c = 1 - cuda::std::abs(rayDir.dot(normal));
    R = R + (1 - R) * c * c * c * c * c;
    return R / (1 + R);
}

__device__ ShaderOutput refraction(ShaderOutput s, Vec normal, double ior, Col col, curandState &randState) {
    bool into = normal.dot(s.d) < 0;
    Vec nl = into ? normal : normal*-1; //normal in direction of ray
    ior = into ? 1/ior : ior; //IOR ratio
    double cost1 = s.d.dot(nl); //cosine of angle of incidence
    double cos2t2 = 1 - ior*ior*(1-cost1*cost1); //cosine of angle of refraction squared

    return ShaderOutput(( s.d*ior - nl*( cost1*ior + cuda::std::sqrt(cuda::std::abs(cos2t2)) ) ).norm(), s.F*col);
}

__device__ ShaderOutput diffuse(ShaderOutput s, Vec normal, Col col, curandState &randState) { //doesn't work
    Vec nl = normal.dot(s.d) > 0 ? normal * -1 : normal;
    double r1 = 2 * M_PI * curand_uniform(&randState);
    double r2 = curand_uniform(&randState);
    Vec u = ((cuda::std::abs(nl.x) > .1) ? Vec(0, 1, 0) : Vec(1, 0, 0)).cross(nl).norm();
    Vec v = nl.cross(u);
    return ShaderOutput(((u * cos(r1) + v * sin(r1)) * cuda::std::sqrt(r2) + nl * cuda::std::sqrt(1 - r2)).norm(), s.F * col);
}

__device__ ShaderOutput glossy(ShaderOutput s, Vec normal, Col col, double rough, curandState &randState) {
    normal = normal.dot(s.d) > 0 ? normal * -1 : normal;
    Vec u = ((cuda::std::abs(normal.x) > .1) ? Vec(0, 1, 0) : Vec(1, 0, 0)).cross(normal).norm();
    Vec v = normal.cross(u);
    double phi = curand_uniform(&randState) * 2 * M_PI;
    Vec w = u * cos(phi) + v * sin(phi);
    
    Vec newRay, newNormal;
    double grad;
    int i;

    for (i = 0; i < RANDOMNESS_ACCURACY; i++) {
        grad = curand_normal(&randState) * rough;
        newNormal = (normal + w*grad).norm();
        newRay = s.d - newNormal * 2 * newNormal.dot(s.d);
        if (newRay.dot(normal) > 0) break;
    }
    if (i == RANDOMNESS_ACCURACY) newRay = s.d - normal * 2 * normal.dot(s.d);

    return ShaderOutput(newRay, s.F * col);
}



__device__ Col radiance(Sphere *spheres, int numSpheres, Vec o, Vec d, Col background, curandState &randState) {
    double t;
    int id = 0;

    Col L = Col();
    ShaderOutput s(d, Col(1, 1, 1));
    Vec x, n;
    Sphere *sphere;

    for (int depth = 0; depth < MAX_BOUNCES + 1; depth++) {
        if (!intersect(spheres, numSpheres, Ray(o, s.d), t, id)) return L + s.F * background;

        sphere = &spheres[id];

        x = o + s.d * t;
        n = (x - sphere->pos).norm(); //normal to sphere

        L = L + s.F * sphere->material.em;

        s = MIX_SHADERS(
            MIX_SHADERS(
                diffuse(s, n, sphere->material.diff, randState),
                refraction(s, n, sphere->material.ior, sphere->material.diff, randState),
                sphere->material.tr,
                randState
            ),
            glossy(
                s, n, 
                mix(
                    Col(1,1,1), 
                    sphere->material.diff, 
                    sphere->material.metal
                ), 
                sphere->material.rough, randState
            ),
            mix(fresnel(sphere->material.ior, s.d, n), 1, sphere->material.metal),
            randState
        );
        o = x;
        if (max(max(s.F.r, s.F.g), s.F.b) < MIN_BRIGHTNESS) break;
    }
    return L;
}



__global__ void render(Ray cam, Vec cx, Vec cy, Col background, Col *c, Sphere *spheres, int numSpheres, int *complete)
{
    int x = (blockIdx.x % TILE_X) * TILE_WIDTH_X + threadIdx.x % TILE_WIDTH_X;
    int y = (blockIdx.x / TILE_X) * TILE_WIDTH_Y + threadIdx.x / TILE_WIDTH_X;

    if (threadIdx.x == 0 && x == 0) {
        printf("\rRendering: %d%%", (100*++(*complete))/TILE_Y);
    }
    
    if (x >= WIDTH || y >= HEIGHT) return;

    int id = (HEIGHT - y - 1) * WIDTH + x;

    curandState state;
    curand_init((unsigned long long)clock() + id, 0, 0, &state);

    Vec d, d1;
    double phi, r;

    for (int s = 0; s < SAMPLES; s++)
    {
        phi = 2*M_PI*curand_uniform(&state);
        r = cuda::std::sqrt(curand_uniform(&state));
        d = (cx * ((curand_uniform(&state) + x) / WIDTH - .5) +
             cy * ((curand_uniform(&state) + y) / HEIGHT - .5) + cam.d)
                .norm();

        d1 = (d + (cx.norm() * cos(phi) + cy.norm() * sin(phi)) * DOF * FD * r);

        c[id] = c[id] + radiance(spheres, numSpheres, cam.o + d1, (d * (FD / d.dot(cam.d)) - d1).norm(), background, state);
    }
    c[id] = c[id] * (1. / SAMPLES);

    c[id].r = toInt(c[id].r);
    c[id].g = toInt(c[id].g);
    c[id].b = toInt(c[id].b);
}



int main(int argc, char *argv[])
{
    Ray cam(Vec(0,0,0), Vec(0, 0, 1));
    Vec camRight = Vec(1,0,0);
    double f = 2 * std::tan(FOV * M_PI / 360);
    Vec cx = camRight*(f * WIDTH/HEIGHT);
    Vec cy = (cam.d.cross(cx)).norm() * f;
    Col background = Col();

    Sphere spheres[] = {
        Sphere(1,  Vec(   0,     0,   3.8), Material(Col(  1,.64,.57),     Col(), 0, 0, 1.45, 0)),
        Sphere(2,  Vec( 3.9,     1,   6.6), Material(Col(  1,.71,.91),     Col(), 1,.2, 1.45, 0)),
        Sphere(1,  Vec(   4,     0,    12), Material(Col(.66,.71,.89),     Col(), 0, 0, 1.45, 0)),
        Sphere(.6, Vec(   1,   -.4,   2.9), Material(Col(1,1,1),           Col(), 0, 0, 1.45, 1)),
        Sphere(.3, Vec( -.4,   -.7,     2), Material(Col(),     Col(12, 6, 0),    0, 1, 1.45, 0)),
        Sphere(.5, Vec(-1.4,   -.5,   3.4), Material(Col(  1,.49,.55),     Col(), 1, 0, 1.45, 0)),
        Sphere(1,  Vec(-2.9,     0,   3.8), Material(Col(.91,.61,.65),     Col(), 0, 1, 1.45, 0)),
        Sphere(1e5,Vec(   0,-1-1e5,     6), Material(           Col(),     Col(), 0,.1, 1.45, 0)),
        Sphere(1e5,Vec(   0, 9+1e5,     6), Material(Col(  1,  1,  1),     Col(), 0, 1, 1.45, 0)),
        Sphere(1e5,Vec(   0,     0,14+1e5), Material(Col(  1,  1,  1),     Col(), 0, 1, 1.45, 0)),
        Sphere(40 ,Vec(   0,    45,     6), Material(     Col(),Col(1.5,1.5,1.8), 0, 1, 1.45, 0))
    };

    int numSpheres = sizeof(spheres) / sizeof(Sphere);

    Col *c;
    Sphere *spheresGPU;
    int *complete;
    CUDA_CALL(cudaMallocManaged(&c, WIDTH * HEIGHT * sizeof(Col)));
    CUDA_CALL(cudaMallocManaged(&spheresGPU, sizeof(spheres)));
    CUDA_CALL(cudaMalloc(&complete, sizeof(int)));

    for (int i = 0; i < numSpheres; i++)
        spheresGPU[i] = spheres[i];

    clock_t startTime = std::clock();
    render<<<TILE_X * TILE_Y, TILE_WIDTH_X * TILE_WIDTH_Y>>>(cam, cx, cy, background, c, spheresGPU, numSpheres, complete);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    printf("\rRendering: 100%%\nTime elapsed: %ds", (int)((std::clock() - startTime)/CLOCKS_PER_SEC));

    std::ofstream file("image.ppm");
    file << "P3\n" << WIDTH << " " << HEIGHT << "\n" << 255 << "\n";
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        file << (int)c[i].r << " " << (int)c[i].g << " " << (int)c[i].b << " ";
    }

    file.close();
    CUDA_CALL(cudaFree(c));
    CUDA_CALL(cudaFree(spheresGPU));
    CUDA_CALL(cudaFree(complete));
}