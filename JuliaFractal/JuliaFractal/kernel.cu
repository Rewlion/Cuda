
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SDL2/SDL.h>

#define SIZE (int)800
#define BITMAP_SIZE (SIZE * SIZE)

struct cuComplex
{
    float r;
    float i;
    __device__ cuComplex(float a, float b)
        : r(a), i(b) 
    {
    }

    __device__ float magnitude2()
    {
        return r * r + i * i;
    }

    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r * a.r - i * a.i, i*a.r + r * a.i);
    }

    __device__ cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

__host__ __device__ inline int get2D(const int x, const int y)
{
    return x + y * SIZE;
}

__device__ int IsJuliaPoint(const int x, const int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(SIZE / 2 - x) / (SIZE / 2);
    float jy = scale * (float)(SIZE / 2 - y) / (SIZE / 2);

    cuComplex c{ -0.8, 0.156 };
    cuComplex a{ jx, jy };

    for (int i = 0; i < 200; ++i)
    {
        a = a * a + c;

        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void CalculateJuliaBitmap(unsigned char* bitmap)
{
    const int x = blockIdx.x;
    const int y = blockIdx.y;

    int p = IsJuliaPoint(x, y);
    bitmap[get2D(x, y)] = p;
}

int main(int argc, char* argv[])
{
    SDL_Window* pWindow = nullptr;
    SDL_Renderer* pRender = nullptr;
    SDL_CreateWindowAndRenderer(SIZE, SIZE, 0, &pWindow, &pRender);
    
    SDL_SetRenderDrawColor(pRender, 0, 0, 0, 255);
    SDL_RenderClear(pRender);

    unsigned char* dev_bitmap;
    cudaMalloc(&dev_bitmap, BITMAP_SIZE);

    dim3 grid(SIZE, SIZE);
    CalculateJuliaBitmap<<<grid,1>>>(dev_bitmap);

    unsigned char* bitmap = (unsigned char*)calloc(BITMAP_SIZE, 1);
    cudaMemcpy(bitmap, dev_bitmap, BITMAP_SIZE, cudaMemcpyDeviceToHost);

    SDL_SetRenderDrawColor(pRender, 0, 255, 100, 255);
    for (int y = 0; y < SIZE; ++y)
        for (int x = 0; x < SIZE; ++x)
        {
            unsigned char bit = bitmap[get2D(x, y)];
            if (bit != 0)
            {
                SDL_RenderDrawPoint(pRender, x, y);
            }
        }
    SDL_RenderPresent(pRender);

    volatile bool f = true;
    while(f) {}

    SDL_Quit();

    return 0;
}