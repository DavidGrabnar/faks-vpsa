#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "FreeImage.h"
#include <CL/cl.h>
#include <errno.h>
#include <unistd.h>

#define error(message) { fprintf(stderr,"Error: %s. (%d)\n", message, errno); exit(1); }

#define MESSAGE_MAX_LENGTH 128
#define FILE_PATH_MAX_LENGTH 128

#define DEFAULT_CLUSTER_COUNT 64
#define DEFAULT_ITERATIONS 10

#define PHI_2D 1.32471795724474602596

#define WORKGROUP_SIZE (512)
#define MAX_SOURCE_SIZE 16384

struct image
{
    char* file_name;
    int width;
    int height;
    int pitch;
    unsigned char *content;
};

struct config
{
    char **image_sizes;
    int image_size_count;

    int *modes; // 1 - cpu serial, 2 - cpu parallel, 3 - gpu global, 4 - gpu local
    int mode_count;

    int cluster_count;
    int iteration_count;

    struct image * image;
};

void get_image(char* file_name, struct image * image);
void save_image(struct config * config, int mode);
void clear_image(struct config * config);

void get_input_path(char* file_name, char * file_path);
void get_output_path(struct config * config, int mode, char * file_path);

void compress(struct config * config);

void run_compress(struct config * config, int mode_index);

// cpu related
void gamma(int d);
void initialize_centroids(unsigned char *image, unsigned char *centroids, int centroid_count);

void compress_cpu_serial(unsigned char *image_in, int size);
void compress_cpu_parallel(unsigned char *image_in, int size);
int distance(unsigned char *a, unsigned char *b, int a_index, int b_index);


// gpu related
void compress_gpu(unsigned char* image_in, int width, int height, int pitch);


int main(int argc, char *argv[])
{
    // init seed
    srand(time(NULL));

    struct config config;
    struct image image;
    config.image = &image;

    // config
    config.image_sizes = (char* []){"640x480", "800x600", "1600x900", "1920x1080", "3840x2160"};
    config.image_size_count = 5;

    config.modes = (int []){1, 2, 3, 4};
    config.mode_count = 4;

    // arguments
    config.cluster_count = DEFAULT_CLUSTER_COUNT;
    config.iteration_count = DEFAULT_ITERATIONS;

    if (argc > 1)
    {
        config.cluster_count = atoi(argv[1]);
    }
    if (argc > 2)
    {
        config.iteration_count = atoi(argv[2]);
    }

    // apply magic
    compress(&config);

    // clear image
    clear_image(&config);

    return 0;
}

void get_image(char* file_name, struct image * image)
{
    char file_path[FILE_PATH_MAX_LENGTH];
    get_input_path(file_name, file_path);

    // open and prepare image
    FIBITMAP *image_bitmap = FreeImage_Load(FIF_PNG, file_path, 0);
    FIBITMAP *image_bitmap32 = FreeImage_ConvertTo32Bits(image_bitmap);

    int width = FreeImage_GetWidth(image_bitmap32);
    int height = FreeImage_GetHeight(image_bitmap32);
    int pitch = FreeImage_GetPitch(image_bitmap32);

    unsigned char *content = (unsigned char *)calloc(height * pitch, sizeof(unsigned char));

    FreeImage_ConvertToRawBits(content, image_bitmap32, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    FreeImage_Unload(image_bitmap32);
    FreeImage_Unload(image_bitmap);

    image->file_name = file_name;
    image->width = width;
    image->height = height;
    image->pitch = pitch;
    image->content = content;
}

void save_image(struct config * config, int mode)
{
    char file_path[FILE_PATH_MAX_LENGTH];
    get_output_path(config, mode, file_path);

    FIBITMAP *dst = FreeImage_ConvertFromRawBits(config->image->content, config->image->width, config->image->height, config->image->pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
    FreeImage_Save(FIF_PNG, dst, file_path, 0);
}

void clear_image(struct config * config)
{
    free(config->image->content);
}

void get_input_path(char* file_name, char * file_path)
{
    sprintf(file_path, "original/%s.png", file_name);
}

void get_output_path(struct config * config, int mode, char * file_path)
{
    char* mode_label;
    char message[MESSAGE_MAX_LENGTH];

    switch (mode) {
        case 1:
            mode_label = "cpu_serial";
            break;
        case 2:
            mode_label = "cpu_openmp";
            break;
        case 3:
            mode_label = "gpu_global";
            break;
        case 4:
            mode_label = "gpu_local";
            break;
        default:
            sprintf(message, "Invalid mode '%d' in get_output_folder. Should be in range [1, 4]", mode);
            error(message);
    }

    sprintf(file_path, "compressed/%s/%d-%d-%s.png", config->image->file_name, config->cluster_count, config->iteration_count, mode_label);
}

void compress(struct config * config)
{
    for (int image_size_index = 0; image_size_index < config->image_size_count; image_size_index++) {
        // get image
        get_image(config->image_sizes[image_size_index], config->image);

        // TODO refactor - extract to a separate method
        // store content to separate buffer
        // and restore the buffer when making a new compression, otherwise each mode will make compression on compressed image
        unsigned int size = config->image->height * config->image->pitch;
        unsigned char *original_content = (unsigned char *)calloc(size, sizeof(unsigned char));
        memcpy(original_content, config->image->content, size);

        for (int mode_index = 0; mode_index < config->mode_count; mode_index++) {
            // compress
            run_compress(config, mode_index);

            // save image
            save_image(config, config->modes[mode_index]);

            // restore original content
            memcpy(config->image->content, original_content, size);
        }
    }
}

void run_compress(struct config * config, int mode_index)
{
    char message[MESSAGE_MAX_LENGTH];
    switch (config->modes[mode_index]) {
        case 1:
            compress_cpu_serial(config->image->content, config->image->width * config->image->height);
            break;
        case 2:
            compress_cpu_parallel(config->image->content, config->image->width * config->image->height);
            break;
        case 3:
            printf("TODO: Compress GPU global not implemented yet. Skipping\n");
            // compress_gpu(config->image->content, config->image->width, config->image->height, config->image->pitch);
            break;
        case 4:
            printf("TODO: Compress GPU local not implemented yet. Skipping\n");
            break;
        default:
        sprintf(message, "Invalid mode %d. Mode should be in range [1, 4]", config->modes[0]);
        error(message);
    }
}

void gamma(int d)
{
    float x = 1f;
    for(int i = 0; i < 20; i++) {

    }
}

// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
// https://math.stackexchange.com/questions/2186861/how-can-we-effectively-generate-a-set-of-evenly-spaced-points-in-a-2d-region
void initialize_centroids(unsigned char *image, unsigned char *centroids, int centroid_count)
{
    for (int i = 0; i < centroid_count; i++)
    {
        int r = rand() % size;


        centroids[i * 3] = image[r * 4];
        centroids[i * 3 + 1] = image[r * 4 + 1];
        centroids[i * 3 + 2] = image[r * 4 + 2];
    }
}

void compress_cpu_serial(unsigned char *image_in, int size)
{
    int centroid_count = 64;
    int iterations = 10;
    unsigned char *centroids = (unsigned char *)malloc(centroid_count * 3 * sizeof(unsigned char));
    int *cluster_indices = (int *)malloc(size * sizeof(int));

    // initialize centroids
    for (int i = 0; i < centroid_count; i++)
    {
        int r = rand() % size;
        centroids[i * 3] = image_in[r * 4];
        centroids[i * 3 + 1] = image_in[r * 4 + 1];
        centroids[i * 3 + 2] = image_in[r * 4 + 2];
    }

    for (int i = 0; i < iterations; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int min_index = 0;
            int min_distance = distance(image_in, centroids, j, 0);
            for (int k = 1; k < centroid_count; k++)
            {
                int d = distance(image_in, centroids, j, k);
                if (d < min_distance)
                {
                    min_index = k;
                    min_distance = d;
                }
            }
            cluster_indices[j] = min_index;
        }
        for (int j = 0; j < centroid_count; j++)
        {
            long sum[3] = {0, 0, 0};
            int count = 0;
            for (int k = 0; k < size; k++)
            {
                if (cluster_indices[k] != j)
                {
                    continue;
                }
                sum[0] += image_in[k * 4];
                sum[1] += image_in[k * 4 + 1];
                sum[2] += image_in[k * 4 + 2];
                count++;
            }
            if (count == 0)
            {
                continue;
            }
            centroids[j * 3] = sum[0] / count;
            centroids[j * 3 + 1] = sum[1] / count;
            centroids[j * 3 + 2] = sum[2] / count;
        }
    }

    // update output TODO reuse input ???
    for (int i = 0; i < size; i++)
    {
        int index = cluster_indices[i];
        image_in[i * 4] = centroids[index * 3];
        image_in[i * 4 + 1] = centroids[index * 3 + 1];
        image_in[i * 4 + 2] = centroids[index * 3 + 2];
    }

    free(centroids);
    free(cluster_indices);
}

void compress_cpu_parallel(unsigned char *image_in, int size)
{
    omp_set_num_threads(4);

    int centroid_count = 64;
    int iterations = 10;
    unsigned char *centroids = (unsigned char *)malloc(centroid_count * 3 * sizeof(unsigned char));
    int *cluster_indices = (int *)malloc(size * sizeof(int));

#pragma omp parallel
    {
// initialize centroids
#pragma omp for schedule(static, 8)
        for (int i = 0; i < centroid_count; i++)
        {
            int r = rand() % size;
            centroids[i * 3] = image_in[r * 4];
            centroids[i * 3 + 1] = image_in[r * 4 + 1];
            centroids[i * 3 + 2] = image_in[r * 4 + 2];
        }

        for (int i = 0; i < iterations; i++)
        {
#pragma omp for schedule(static, 1024)
            for (int j = 0; j < size; j++)
            {
                int min_index = 0;
                int min_distance = distance(image_in, centroids, j, 0);
                for (int k = 1; k < centroid_count; k++)
                {
                    int d = distance(image_in, centroids, j, k);
                    if (d < min_distance)
                    {
                        min_index = k;
                        min_distance = d;
                    }
                }
                cluster_indices[j] = min_index;
            }

#pragma omp for schedule(static, 8)
            for (int j = 0; j < centroid_count; j++)
            {
                long sum[3] = {0, 0, 0};
                int count = 0;
                for (int k = 0; k < size; k++)
                {
                    if (cluster_indices[k] != j)
                    {
                        continue;
                    }
                    sum[0] += image_in[k * 4];
                    sum[1] += image_in[k * 4 + 1];
                    sum[2] += image_in[k * 4 + 2];
                    count++;
                }
                if (count == 0)
                {
                    continue;
                }
                centroids[j * 3] = sum[0] / count;
                centroids[j * 3 + 1] = sum[1] / count;
                centroids[j * 3 + 2] = sum[2] / count;
            }
        }

        // update output TODO reuse input ???
#pragma omp for schedule(static, 1024)
        for (int i = 0; i < size; i++)
        {
            int index = cluster_indices[i];
            image_in[i * 4] = centroids[index * 3];
            image_in[i * 4 + 1] = centroids[index * 3 + 1];
            image_in[i * 4 + 2] = centroids[index * 3 + 2];
        }
    }

    free(centroids);
    free(cluster_indices);
}

int distance(unsigned char *a, unsigned char *b, int a_index, int b_index)
{
    return sqrt(
        pow(a[a_index * 4] - b[b_index * 3], 2) + pow(a[a_index * 4 + 1] - b[b_index * 3 + 1], 2) + pow(a[a_index * 4 + 2] - b[b_index * 3 + 2], 2));
}

void compress_gpu(unsigned char* image_in, int width, int height, int pitch)
{
    int centroid_count = 64;
    int iterations = 10;
    unsigned char *centroids = (unsigned char *)malloc(centroid_count * 3 * sizeof(unsigned char));
    int *cluster_indices = (int *)malloc(width * height * sizeof(int));

    // initialize centroids
    for (int i = 0; i < centroid_count; i++)
    {
        int r = rand() % (width * height);
        centroids[i * 3] = image_in[r * 4];
        centroids[i * 3 + 1] = image_in[r * 4 + 1];
        centroids[i * 3 + 2] = image_in[r * 4 + 2];
    }

    int pitch_image_size = pitch * height;
    int image_size = width * height;

    char ch;
    int i;
    cl_int ret;

	// clock_t file_time;
	// file_time = clock();

    // Branje datoteke
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("kernelCompress.cl", "r");
    if (!fp)
    {
        fprintf(stderr, ":-(#\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);

	// file_time = clock() - file_time;
	// printf("Elapsed file time: %f seconds\n", ((double)file_time)/CLOCKS_PER_SEC);

	// clock_t gpu_setup_time;
	// gpu_setup_time = clock();

    // Podatki o platformi
    cl_platform_id platform_id[10];
    cl_uint ret_num_platforms;
    char *buf;
    size_t buf_len;
    ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
    // max. "stevilo platform, kazalec na platforme, dejansko "stevilo platform

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

    // Delitev dela
    size_t local_item_size = WORKGROUP_SIZE;
    size_t num_groups = ((image_size - 1) / local_item_size + 1);
    size_t global_item_size = num_groups * local_item_size;

	// clock_t copy_image_time;
	// copy_image_time = clock();

    // Alokacija pomnilnika na napravi
    cl_mem image_in_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             pitch_image_size * sizeof(unsigned char), image_in, &ret);

	// copy_image_time = clock() - copy_image_time;
	// printf("Elapsed copy image time: %f seconds\n", ((double)copy_image_time)/CLOCKS_PER_SEC);

    cl_mem centroids_obj = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                          centroid_count * 3 * sizeof(unsigned char), centroids, &ret);

    cl_mem cluster_indices_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                            width * height * sizeof(int), NULL, &ret);

    // Priprava programa
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                                   NULL, &ret);
    // kontekst, "stevilo kazalcev na kodo, kazalci na kodo,
    // stringi so NULL terminated, napaka

    // Prevajanje
    ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);
    // program, "stevilo naprav, lista naprav, opcije pri prevajanju,
    // kazalec na funkcijo, uporabni"ski argumenti

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

    // "s"cepec: priprava objekta
    cl_kernel kernel = clCreateKernel(program, "associate_clusters", &ret);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_in_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&centroids_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&cluster_indices_obj);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&image_size);
    ret |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&centroid_count);

    cl_kernel kernel2 = clCreateKernel(program, "update_centroids", &ret);

    ret = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&image_in_mem_obj);
    ret = clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&centroids_obj);
    ret = clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void *)&cluster_indices_obj);
    ret |= clSetKernelArg(kernel2, 3, sizeof(cl_int), (void *)&image_size);
    ret |= clSetKernelArg(kernel2, 4, sizeof(cl_int), (void *)&centroid_count);


    for (int i = 0; i < iterations; i++) {
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                    &global_item_size, &local_item_size, 0, NULL, NULL);

        ret = clEnqueueNDRangeKernel(command_queue, kernel2, 1, NULL,
                                    &global_item_size, &local_item_size, 0, NULL, NULL);
    }


    ret = clEnqueueReadBuffer(command_queue, cluster_indices_obj, CL_TRUE, 0,
                              width * height * sizeof(int), cluster_indices, 0, NULL, NULL);

    ret = clEnqueueReadBuffer(command_queue, centroids_obj, CL_TRUE, 0,
                              centroid_count * 3 * sizeof(unsigned char), centroids, 0, NULL, NULL);


    // "ci"s"cenje
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(image_in_mem_obj);
    ret = clReleaseMemObject(centroids_obj);
    ret = clReleaseMemObject(cluster_indices_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    // update output TODO reuse input ???
    for (int i = 0; i < width * height; i++)
    {
        int index = cluster_indices[i];
        image_in[i * 4] = centroids[index * 3];
        image_in[i * 4 + 1] = centroids[index * 3 + 1];
        image_in[i * 4 + 2] = centroids[index * 3 + 2];
    }

    free(centroids);
    free(cluster_indices);

	//clear_time = clock() - clear_time;
	//printf("Elapsed clear time: %f seconds\n", ((double)clear_time)/CLOCKS_PER_SEC);
}