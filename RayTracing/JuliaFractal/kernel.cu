
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SDL2/SDL.h>

#include <assert.h>
#include <cstdlib>
#include <ctime>

#define SIZE (int)1024
#define BITMAP_SIZE (SIZE * SIZE)

#define SPHERES_COUNT (int)20
#define INF (float)2e10f;

struct Sphere
{
    float r, g, b;
    float radius;
    float x, y, z;

    __device__ __host__ float hit(float ox, float oy, float* n)
    {
        const float dx = ox - x;
        const float dy = oy - y;

        const float r2 = radius * radius;
        const float l2 = dx * dx + dy * dy;
        
        if (l2 < r2)
        {
            float dz = sqrtf(r2 - l2);
            *n = dz / sqrtf(r2);

            return dz + z;
        }
        else
            return -INF;
    }
};

struct Pixel
{
    unsigned char r, g, b;
};

__device__ __host__ inline int get2D(const int x, const int y)
{
    return x + y * SIZE;
}

__constant__ Sphere spheres_dev[SPHERES_COUNT];

__global__ void kernel(Pixel* bitmap)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const float ox = x - SIZE / 2;
    const float oy = y - SIZE / 2;

    float r = 0, g = 0, b = 0;
    float maxZ = -INF;

    for (int i = 0; i < SPHERES_COUNT; ++i)
    {
        float n;
        float t = spheres_dev[i].hit(ox, oy, &n);
        if (t > maxZ)
        {
            float fscale = n;
            r = spheres_dev[i].r * fscale;
            g = spheres_dev[i].g * fscale;
            b = spheres_dev[i].b * fscale;
            maxZ = t;
        }
    }

    Pixel* p = &bitmap[get2D(x, y)];
    p->r = (int)(r * 255);
    p->g = (int)(g * 255);
    p->b = (int)(b * 255);
}

int main(int argc, char* argv[])
{
    SDL_Window* pWindow = nullptr;
    SDL_Renderer* pRender = nullptr;
    SDL_CreateWindowAndRenderer(SIZE, SIZE, 0, &pWindow, &pRender);
    
    Pixel* bitmap = (Pixel*)calloc(BITMAP_SIZE, sizeof(Pixel));
    Pixel* bitmap_dev;
    assert(cudaMalloc(&bitmap_dev, BITMAP_SIZE * sizeof(Pixel)) == 0);


    srand(time(NULL));
    Sphere* spheres = (Sphere*)calloc(SPHERES_COUNT, sizeof(*spheres));
    for (int i = 0; i < SPHERES_COUNT; ++i)
    {
        spheres[i].r = float(rand()) / (float)RAND_MAX;
        spheres[i].g = float(rand()) / (float)RAND_MAX;
        spheres[i].b = float(rand()) / (float)RAND_MAX;
        spheres[i].x = ((float)rand() / (float)RAND_MAX) * 1000.0f - 500.0f;
        spheres[i].y = ((float)rand() / (float)RAND_MAX) * 1000.0f - 500.0f;
        spheres[i].z = ((float)rand() / (float)RAND_MAX) * 1000.0f - 500.0f;
        spheres[i].radius = ((float)rand() / (float)RAND_MAX) * 100.0f + 20.0f;
    }
    assert(cudaMemcpyToSymbol(spheres_dev, spheres, SPHERES_COUNT * sizeof(Sphere)) == 0);

    dim3 grid(SIZE/16, SIZE/16);
    dim3 threads(16, 16);
    kernel<<<grid, threads>>>(bitmap_dev);
    assert(cudaMemcpy(bitmap, bitmap_dev, BITMAP_SIZE * sizeof(Pixel), cudaMemcpyDeviceToHost) == 0);

    for(int y = 0; y < SIZE; ++y)
        for (int x = 0; x < SIZE; ++x)
        {
            const int li = get2D(x, y);
            Pixel* p = &bitmap[li];
            SDL_SetRenderDrawColor(pRender, p->r, p->g, p->b, 255);
            SDL_RenderDrawPoint(pRender, x, y);
        }

    SDL_RenderPresent(pRender);

    volatile bool f = true;
    while(f) {}

    SDL_Quit();

    return 0;
}