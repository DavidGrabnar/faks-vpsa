#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "FreeImage.h"
#include <CL/cl.h>
#include <string.h>

#define WORKGROUP_SIZE (512)
#define MAX_SOURCE_SIZE 16384

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

void sobelGPU(unsigned char *image_in, unsigned char *image_out, int width, int height)
{
    int image_size = width * height;

    char ch;
    int i;
    cl_int ret;

    // Branje datoteke
    FILE *fp;
    char *source_str;
    size_t source_size;

    printf("Reading CL\n");
    fp = fopen("kernelSobel.cl", "r");
    if (!fp)
    {
        fprintf(stderr, ":-(#\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);

    printf("Reading platform info\n");
    // Podatki o platformi
    cl_platform_id platform_id[10];
    cl_uint ret_num_platforms;
    char *buf;
    size_t buf_len;
    ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
    // max. "stevilo platform, kazalec na platforme, dejansko "stevilo platform

    printf("Reading device info\n");
    // Podatki o napravi
    cl_device_id device_id[10];
    cl_uint ret_num_devices;
    // Delali bomo s platform_id[0] na GPU
    ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10,
                         device_id, &ret_num_devices);
    // izbrana platforma, tip naprave, koliko naprav nas zanima
    // kazalec na naprave, dejansko "stevilo naprav

    // Kontekst
    cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);
    // kontekst: vklju"cene platforme - NULL je privzeta, "stevilo naprav,
    // kazalci na naprave, kazalec na call-back funkcijo v primeru napake
    // dodatni parametri funkcije, "stevilka napake

    // Ukazna vrsta
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);
    // kontekst, naprava, INORDER/OUTOFORDER, napake

    printf("Split work\n");
    // Delitev dela
    size_t local_item_size = WORKGROUP_SIZE;
    size_t num_groups = ((image_size - 1) / local_item_size + 1);
    size_t global_item_size = num_groups * local_item_size;

    printf("Allocate memory\n");
    // Alokacija pomnilnika na napravi
    cl_mem image_in_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                             image_size * sizeof(unsigned char) * 4, NULL, &ret);

    cl_mem image_out_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                              image_size * sizeof(unsigned char) * 4, NULL, &ret);

    printf("Prepare program\n");
    // Priprava programa
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                                   NULL, &ret);
    // kontekst, "stevilo kazalcev na kodo, kazalci na kodo,
    // stringi so NULL terminated, napaka

    printf("Build program\n");
    // Prevajanje
    ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);
    // program, "stevilo naprav, lista naprav, opcije pri prevajanju,
    // kazalec na funkcijo, uporabni"ski argumenti

    printf("Get build info\n");
    // Log
    size_t build_log_len;
    char *build_log;
    ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
                                0, NULL, &build_log_len);
    // program, "naprava, tip izpisa,
    // maksimalna dol"zina niza, kazalec na niz, dejanska dol"zina niza
    build_log = (char *)malloc(sizeof(char) * (build_log_len + 1));
    ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
                                build_log_len, build_log, NULL);
    printf("%s\n", build_log);
    free(build_log);

    printf("Run\n");
    // "s"cepec: priprava objekta
    cl_kernel kernel = clCreateKernel(program, "sobel_gpu", &ret);
    // program, ime "s"cepca, napaka

    printf("Arguments\n");
    // "s"cepec: argumenti
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_in_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&image_out_mem_obj);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&height);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&width);
    ret |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&image_size);
    // "s"cepec, "stevilka argumenta, velikost podatkov, kazalec na podatke

    // "s"cepec: zagon
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                 &global_item_size, &local_item_size, 0, NULL, NULL);
    // vrsta, "s"cepec, dimenzionalnost, mora biti NULL,
    // kazalec na "stevilo vseh niti, kazalec na lokalno "stevilo niti,
    // dogodki, ki se morajo zgoditi pred klicem

    // Kopiranje rezultatov
    ret = clEnqueueReadBuffer(command_queue, image_out_mem_obj, CL_TRUE, 0,
                              image_size * sizeof(unsigned char) * 4, image_out, 0, NULL, NULL);
    // branje v pomnilnik iz naparave, 0 = offset
    // zadnji trije - dogodki, ki se morajo zgoditi prej

    // "ci"s"cenje
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(image_in_mem_obj);
    ret = clReleaseMemObject(image_out_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
}

int main(int argc, char *argv[])
{
    printf("Start\n");
    //Initialize parameters
    int target_gpu = DEFAULT_TARGET_GPU;
    char *input_path = DEFAULT_INPUT_PATH;
    char *output_path = DEFAULT_OUTPUT_PATH;

    if (argc > 1 && !strcmp(argv[1], "0"))
    {
        target_gpu = 0;
    }
    if (argc > 2)
    {
        input_path = argv[2];
    }
    if (argc > 3)
    {
        output_path = argv[3];
    }
        printf("will load memory\n");

    //Load image from file
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, input_path, 0);
    //Convert it to an 8-bit grayscale image
    FIBITMAP *imageBitmap8 = FreeImage_ConvertTo8Bits(imageBitmap);

    printf("Loaded memory\n");
    //Get image dimensions
    int width = FreeImage_GetWidth(imageBitmap8);
    int height = FreeImage_GetHeight(imageBitmap8);
    int pitch = FreeImage_GetPitch(imageBitmap8);

    printf("Preparing memory\n");
    //Preapare room for a raw data copy of the image
    unsigned char *image_in = (unsigned char *)malloc(height * pitch * sizeof(unsigned char));
    FreeImage_ConvertToRawBits(image_in, imageBitmap8, pitch, 8, 0xFF, 0xFF, 0xFF, TRUE);

    unsigned char *image_out = (unsigned char *)malloc(height * pitch * sizeof(unsigned char));

    //find edges
    if (target_gpu)
    {
        printf("Using GPU\n");
        sobelGPU(image_in, image_out, width, height);
    }
    else
    {
        printf("Using CPU\n");
        sobelCPU(image_in, image_out, width, height);
    }

    //save output image
    FIBITMAP *dst = FreeImage_ConvertFromRawBits(image_out, width, height, pitch,
                                                 8, 0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_Save(FIF_PNG, dst, output_path, 0);

    return 0;
}
