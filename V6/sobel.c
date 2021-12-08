#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "FreeImage.h"

#define DEFAULT_TARGET_GPU 1
#define DEFAULT_INPUT_PATH "input.png"
#define DEFAULT_OUTPUT_PATH "robovi.png"

int getPixel(unsigned char *image, int y, int x, int width, int height)
{
    if (x < 0 || x >= width)
        return 0;
    if (y < 0 || y >= height)
        return 0;
    return image[y * width + x];
}

void sobelCPU(unsigned char *image_in, unsigned char *image_out, int width, int height)
{
    int i, j;
    int Gx, Gy;
    int tempPixel;

    //za vsak piksel v sliki
    for (i = 0; i < height; i++)
        for (j = 0; j < width; j++)
        {
            Gx = -getPixel(image_in, i - 1, j - 1, width, height) - 2 * getPixel(image_in, i - 1, j, width, height) -
                 getPixel(image_in, i - 1, j + 1, width, height) + getPixel(image_in, i + 1, j - 1, width, height) +
                 2 * getPixel(image_in, i + 1, j, width, height) + getPixel(image_in, i + 1, j + 1, width, height);
            Gy = -getPixel(image_in, i - 1, j - 1, width, height) - 2 * getPixel(image_in, i, j - 1, width, height) -
                 getPixel(image_in, i + 1, j - 1, width, height) + getPixel(image_in, i - 1, j + 1, width, height) +
                 2 * getPixel(image_in, i, j + 1, width, height) + getPixel(image_in, i + 1, j + 1, width, height);
            tempPixel = sqrt((float)(Gx * Gx + Gy * Gy));
            if (tempPixel > 255)
                image_out[i * width + j] = 255;
            else
                image_out[i * width + j] = tempPixel;
        }
}

int main(int argc, char *argv[])
{
    //Initialize parameters
    int target_gpu = DEFAULT_TARGET_GPU;
    char* input_path = DEFAULT_INPUT_PATH;
    char* output_path = DEFAULT_OUTPUT_PATH;

	if (argc > 1 && !strcmp(argv[1], "0")) {
		target_gpu = 0;
	}
    if (argc > 2) {
        input_path = argv[2];
    }
    if (argc > 3) {
        output_path = argv[3];
    }

    //Load image from file
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, input_path, 0);
    //Convert it to an 8-bit grayscale image
    FIBITMAP *imageBitmap8 = FreeImage_ConvertTo8Bits(imageBitmap);

    //Get image dimensions
    int width = FreeImage_GetWidth(imageBitmap8);
    int height = FreeImage_GetHeight(imageBitmap8);
    int pitch = FreeImage_GetPitch(imageBitmap8);
    
    //Preapare room for a raw data copy of the image
    unsigned char *image_in = (unsigned char *)malloc(height * pitch * sizeof(unsigned char));
    FreeImage_ConvertToRawBits(image_in, imageBitmap8, pitch, 8, 0xFF, 0xFF, 0xFF, TRUE);

    unsigned char *image_out = (unsigned char *)malloc(height * pitch * sizeof(unsigned char));

    //find edges
    if (target_gpu == 1) {
        sobelCPU(image_in, image_out, width, height);
    } else {
        // sobelGPU(image_in, image_out, width, height);
    }

    //save output image
    FIBITMAP *dst = FreeImage_ConvertFromRawBits(image_out, width, height, pitch,
		8, 0xFF, 0xFF, 0xFF, TRUE);
	FreeImage_Save(FIF_PNG, dst, output_path, 0);

    return 0;
}
