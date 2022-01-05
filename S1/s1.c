#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "FreeImage.h"

#define DEFAULT_INPUT_PATH "640x480.png"
#define DEFAULT_OUTPUT_PATH "640x480_out.png"

void compress_cpu(unsigned char *image_in, unsigned char *image_out, int width, int height, int pitch);
int distance(unsigned char *a, unsigned char *b, int a_index, int b_index);


int main(int argc, char *argv[])
{
    // Initialize random
    srand(time(NULL));

    //Initialize parameters
    char *input_path = DEFAULT_INPUT_PATH;
    char *output_path = DEFAULT_OUTPUT_PATH;

    if (argc > 1)
    {
        input_path = argv[1];
    }
    if (argc > 2)
    {
        output_path = argv[2];
    }

    FIBITMAP *image_bitmap = FreeImage_Load(FIF_PNG, input_path, 0);
    FIBITMAP *image_bitmap32 = FreeImage_ConvertTo32Bits(image_bitmap);

    int width = FreeImage_GetWidth(image_bitmap32);
    int height = FreeImage_GetHeight(image_bitmap32);
    int pitch = FreeImage_GetPitch(image_bitmap32);

    unsigned char *image_in = (unsigned char *)calloc(height * pitch, sizeof(unsigned char));

    FreeImage_ConvertToRawBits(image_in, image_bitmap32, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    FreeImage_Unload(image_bitmap32);
    FreeImage_Unload(image_bitmap);

    unsigned char *image_out = (unsigned char *)calloc(height * pitch, sizeof(unsigned char));

    compress_cpu(image_in, image_out, width, height, pitch);

    FIBITMAP *dst = FreeImage_ConvertFromRawBits(image_out, width, height, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
    FreeImage_Save(FIF_PNG, dst, output_path, 0);
    free(image_in);
    free(image_out);

    return 0;
}

void compress_cpu(unsigned char *image_in, unsigned char *image_out, int width, int height, int pitch)
{
    int depth = pitch / width;
    int samples = 64;
    int iterations = 10;
    unsigned char *clusters = (unsigned char *)malloc(samples * depth * sizeof(unsigned char));
    int *indices = (int *)malloc(width * height * sizeof(int));

    for (int i = 0; i < samples; i++) {
        int r = rand() % (width * height);
        clusters[i * 4] = image_in[r * 4];
        clusters[i * 4 + 1] = image_in[r * 4 + 1];
        clusters[i * 4 + 2] = image_in[r * 4 + 2];
    }
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < width * height; j++) {
            int min_index = 0;
            int min_distance = distance(image_in, clusters, j, 0); 
            for (int k = 1; k < samples; k++) {
                int d = distance(image_in, clusters, j, k);
                if (d < min_distance) {
                    min_index = k;
                    min_distance = d;
                }
            }
            indices[j] = min_index;
        }
        for (int j = 0; j < samples; j++) {
            int sum[3];
            int count = 0;
            for (int k = 0; k < width * height; k++) {
                if (indices[k] != j) {
                    continue;
                }
                sum[0] += image_in[k * 4];
                sum[1] += image_in[k * 4 + 1];
                sum[2] += image_in[k * 4 + 2];
                count++;
            }
            if (count == 0) {
                continue;
            }
            clusters[i * 4] = sum[0] / count;
            clusters[i * 4 + 1] = sum[1] / count;
            clusters[i * 4 + 2] = sum[2] / count;
        }
    }

    for (int i = 0; i < width * height; i++) {
        int index = indices[i];
        image_out[i * 4] = clusters[index * 4];
        image_out[i * 4 + 1] = clusters[index * 4 + 1];
        image_out[i * 4 + 2] = clusters[index * 4 + 2];
    }
    free(clusters);
    free(indices);
}

int distance(unsigned char *a, unsigned char *b, int a_index, int b_index)
{
    return sqrt(
        pow(a[a_index * 4] - b[b_index * 4], 2)
        + pow(a[a_index * 4 + 1] - b[b_index * 4 + 1], 2)
        + pow(a[a_index * 4 + 2] - b[b_index * 4 + 2], 2)
    );
}