#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "FreeImage.h"
#include <CL/cl.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#define error(message) { fprintf(stderr,"Error: %s. (%d)\n", message, errno); exit(1); }

#define cl_error_check(status, message) { if(status != CL_SUCCESS) error(message); }

#define MESSAGE_MAX_LENGTH 128
#define FILE_PATH_MAX_LENGTH 128

#define DEFAULT_CLUSTER_COUNT 64
#define DEFAULT_ITERATIONS 10

#define DEFAULT_THREAD_COUNT 4

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

    int thread_count;

    struct image * image;
};

void get_image(char* file_path, char * file_name, struct image * image);
void save_image(struct config * config, int mode);
void clear_image(struct config * config);

void get_input_path(char* file_name, char * file_path);
void get_output_path(struct config * config, int mode, char * file_name, char * file_path);

void get_mode_label(char* label, int mode);

double** compress(struct config * config);
void print_durations(struct config * config, double ** durations);
void compare_results(struct config * config);
int pixel_equals(struct image * source, struct image * destination, int index);

double run_compress(struct config * config, int mode_index);

// cpu related
double compress_cpu(struct config * config);
void initialize_centroids(struct config * config, unsigned char *centroids);
int distance(unsigned char *a, unsigned char *b, int a_index, int b_index);


// gpu related
double compress_gpu(struct config * config);
void get_cl_source(char* source);
cl_device_id get_device();


int main(int argc, char *argv[])
{
    printf("Welcome to compressor 9000\n");
    
	clock_t total_time;
	total_time = clock();

    // init seed
    srand(time(NULL));

    struct config config;
    struct image image;
    config.image = &image;

    // config
    //config.image_sizes = (char* []){"640x480"};
    config.image_sizes = (char* []){"640x480", "800x600", "1600x900", "1920x1080", "3840x2160"};
    //config.image_size_count = 1;
    config.image_size_count = 5;

    config.modes = (int []){1, 2, 3, 4};
    config.mode_count = 4;

    // arguments
    config.cluster_count = DEFAULT_CLUSTER_COUNT;
    config.iteration_count = DEFAULT_ITERATIONS;
    config.thread_count = DEFAULT_THREAD_COUNT;

    if (argc > 1)
    {
        config.cluster_count = atoi(argv[1]);
    }
    if (argc > 2)
    {
        config.iteration_count = atoi(argv[2]);
    }
    if (argc > 3)
    {
        config.thread_count = atoi(argv[3]);
    }

    // apply magic
    double ** durations = compress(&config);

    print_durations(&config, durations);

    free(durations);

    compare_results(&config);


	total_time = clock() - total_time;
    printf("Finished program. Took: %f seconds\n", ((double)total_time) / CLOCKS_PER_SEC);

    printf("Goodbye\n");
    return 0;
}

void get_image(char* file_path, char* file_name, struct image * image)
{
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
    get_output_path(config, mode, config->image->file_name, file_path);

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

void get_output_path(struct config * config, int mode, char * file_name, char * file_path)
{
    char message[MESSAGE_MAX_LENGTH];
    char* mode_label;
    switch (mode) {
        case 1:
            mode_label = "cpu_serial";
            break;
        case 2:
            mode_label = "cpu_parallel";
            break;
        case 3:
            mode_label = "gpu_global";
            break;
        case 4:
            mode_label = "gpu_local";
            break;
        default:
            sprintf(message, "Invalid mode '%d' in get_output_path. Should be in range [1, 4]", mode);
            error(message);
    }

    sprintf(file_path, "compressed/%s/%d-%d-%s.png", file_name, config->cluster_count, config->iteration_count, mode_label);
}

// TODO this is not working ?
void get_mode_label(char* label, int mode)
{
    char message[MESSAGE_MAX_LENGTH];

    label = malloc(32);
    switch (mode) {
        case 1:
            strcpy(label, "cpu_serial");
            //label = "cpu_serial";
            break;
        case 2:
            strcpy(label, "cpu_parallel");
            //label = "cpu_parallel";
            break;
        case 3:
            strcpy(label, "gpu_global");
            //label = "gpu_global";
            break;
        case 4:
            strcpy(label, "gpu_local");
            //label = "gpu_local";
            break;
        default:
            sprintf(message, "Invalid mode '%d' in get_mode_label. Should be in range [1, 4]", mode);
            error(message);
    }

    printf("test %d - %s\n", mode, label);
}

double** compress(struct config * config)
{
    printf("Starting compress\n");
    
	clock_t total_time;
	total_time = clock();

    double **durations = (double **)calloc(config->image_size_count, sizeof(double *));

    for (int image_size_index = 0; image_size_index < config->image_size_count; image_size_index++) {
        printf("Progress: image size %d/%d\n", image_size_index + 1, config->image_size_count);
        durations[image_size_index] = (double *)calloc(config->mode_count, sizeof(double));
        // get image
        char file_path[FILE_PATH_MAX_LENGTH];
        get_input_path(config->image_sizes[image_size_index], file_path);
        get_image(file_path, config->image_sizes[image_size_index], config->image);

        // store content to a separate buffer
        unsigned int size = config->image->height * config->image->pitch;
        unsigned char *original_content = (unsigned char *)calloc(size, sizeof(unsigned char));
        memcpy(original_content, config->image->content, size);

        for (int mode_index = 0; mode_index < config->mode_count; mode_index++) {
            printf("Progress: compress mode %d/%d\n", mode_index + 1, config->mode_count);
            // compress
            durations[image_size_index][mode_index] = run_compress(config, mode_index);

            // save image
            save_image(config, config->modes[mode_index]);

            // restore original content
            memcpy(config->image->content, original_content, size);
        }
    }

	total_time = clock() - total_time;
    printf("Finished compress. Took: %f seconds\n", ((double)total_time) / CLOCKS_PER_SEC);
    clear_image(config);

    return durations;
}

void print_durations(struct config * config, double ** durations)
{
    printf("---- Results ----\n");
    printf("Threads for OMP: %d\n", config->thread_count);
    printf("Cluster count: %d\n", config->cluster_count);
    printf("Iterations: %d\n", config->iteration_count);
    printf("-----------------\n");
    printf("      Image     ");
    for (int mode_index = 0; mode_index < config->mode_count; mode_index++) {
        char* mode_label;
        switch (config->modes[mode_index]) {
            case 1:
                mode_label = "cpu_serial";
                break;
            case 2:
                mode_label = "cpu_parallel";
                break;
            case 3:
                mode_label = "gpu_global";
                break;
            case 4:
                mode_label = "gpu_local";
                break;
            default:
                mode_label = "unknown";
        }
        printf("|%13s ", mode_label);
    }
    printf("\n");

    for (int image_size_index = 0; image_size_index < config->image_size_count; image_size_index++) {
        printf("%15s ", config->image_sizes[image_size_index]);
        for (int mode_index = 0; mode_index < config->mode_count; mode_index++) {
            printf("| %12.6f ", durations[image_size_index][mode_index]);
        }
        printf("\n");
    }
    printf("-----------------\n");
}

void compare_results(struct config * config)
{
    printf("---- Comparing serial to other ----\n");
    for (int image_size_index = 0; image_size_index < config->image_size_count; image_size_index++)
    {
        printf("Comparing image '%s'\n", config->image_sizes[image_size_index]);

        struct image image_serial;
        char file_path_serial[FILE_PATH_MAX_LENGTH];
        get_output_path(config, config->modes[0], config->image_sizes[image_size_index], file_path_serial);
        get_image(file_path_serial, config->image_sizes[image_size_index], &image_serial);

        int size = image_serial.width * image_serial.height;

        // starting at 1 to skip serial since we are comparting to it
        // TODO subtracted 2 to skip GPU until its working
        for (int mode_index = 1; mode_index < config->mode_count - 1; mode_index++)
        {
            char* mode_label;
            switch (config->modes[mode_index]) {
                case 1:
                    mode_label = "cpu_serial";
                    break;
                case 2:
                    mode_label = "cpu_parallel";
                    break;
                case 3:
                    mode_label = "gpu_global";
                    break;
                case 4:
                    mode_label = "gpu_local";
                    break;
                default:
                    mode_label = "unknown";
            }
            printf("Comparing serial to '%s'\n", mode_label);

            struct image image_mode;
            char file_path_mode[FILE_PATH_MAX_LENGTH];
            get_output_path(config, config->modes[mode_index], config->image_sizes[image_size_index], file_path_mode);
            get_image(file_path_mode, config->image_sizes[image_size_index], &image_mode);

            int diff_count = 0;
            for(int index = 0; index < size; index++)
            {
                if (!pixel_equals(&image_serial, &image_mode, index))
                {
                    int x = index % image_serial.width;
                    int y = index / image_serial.height;
                    //printf("Difference: %s - %s @(%d, %d): (%d, %d, %d) -> (%d, %d, %d)\n", 
                    //    config->image_sizes[image_size_index], mode_label, x, y,
                    //    image_serial.content[index * 4], image_serial.content[index * 4 + 1], image_serial.content[index * 4 + 2],
                    //    image_mode.content[index * 4], image_mode.content[index * 4 + 1], image_mode.content[index * 4 + 2]
                    //);
                    diff_count++;
                }
            }
            if (diff_count == 0) {
                printf("Images are equal\n");
            } else {
                printf("Found %d differences in images\n", diff_count);
            }
        }
    }
}

int pixel_equals(struct image * source, struct image * destination, int index)
{
    return (
        source->content[index * 4] == destination->content[index * 4]
        && source->content[index * 4 + 1] == destination->content[index * 4 + 1]
        && source->content[index * 4 + 2] == destination->content[index * 4 + 2]
    );
}

double run_compress(struct config * config, int mode_index)
{
    char message[MESSAGE_MAX_LENGTH];
    switch (config->modes[mode_index]) {
        case 1:
            omp_set_num_threads(1);
            return compress_cpu(config);
        case 2:
            omp_set_num_threads(config->thread_count);
            return compress_cpu(config);
        case 3:
            return compress_gpu(config);
            //printf("TODO: Compress GPU local not implemented yet. Skipping\n");
            return 0;
        case 4:
            printf("TODO: Compress GPU local not implemented yet. Skipping\n");
            return 0;
        default:
            sprintf(message, "Invalid mode %d. Mode should be in range [1, 4]", config->modes[0]);
            error(message);
    }
}

// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
// https://math.stackexchange.com/questions/2186861/how-can-we-effectively-generate-a-set-of-evenly-spaced-points-in-a-2d-region
void initialize_centroids(struct config * config, unsigned char *centroids)
{
    double alpha[2];
    int sizes[2];
    sizes[0] = config->image->width;
    sizes[1] = config->image->height;

    for (int i = 0; i < 2; i++) {
        alpha[i] = fmod(pow(1.0 / PHI_2D, i + 1), 1.0);
    }

    double seed = 0.5;

#pragma omp for schedule(static, 8)
    for (int i = 0; i < config->cluster_count; i++)
    {
        int x = fmod(seed + alpha[0] * (i + 1), 1) * sizes[0];
        int y = fmod(seed + alpha[1] * (i + 1), 1) * sizes[1];

        int index = y * config->image->width + x;

        centroids[i * 3] = config->image->content[index * 4];
        centroids[i * 3 + 1] = config->image->content[index * 4 + 1];
        centroids[i * 3 + 2] = config->image->content[index * 4 + 2];
    }
}

double compress_cpu(struct config * config)
{
    int size = config->image->width * config->image->height;
    unsigned char *centroids = (unsigned char *)malloc(config->cluster_count * 3 * sizeof(unsigned char));
    int *cluster_indices = (int *)malloc(size * sizeof(int));
    double start_time;

#pragma omp parallel
    {
        start_time = omp_get_wtime();
        initialize_centroids(config, centroids);

        for (int i = 0; i < config->iteration_count; i++)
        {
#pragma omp for schedule(static, 1024)
            for (int j = 0; j < size; j++)
            {
                int min_index = 0;
                int min_distance = distance(config->image->content, centroids, j, 0);
                for (int k = 1; k < config->cluster_count; k++)
                {
                    int d = distance(config->image->content, centroids, j, k);
                    if (d < min_distance)
                    {
                        min_index = k;
                        min_distance = d;
                    }
                }
                cluster_indices[j] = min_index;
            }
            // TODO rewrite with 2D[clusters][dimensions] sum and 1D[clusters] count array, requires only 1xsize loop + 1xclusters loop
#pragma omp for schedule(static, 8)
            for (int j = 0; j < config->cluster_count; j++)
            {
                long sum[3] = {0, 0, 0};
                int count = 0;
                for (int k = 0; k < size; k++)
                {
                    if (cluster_indices[k] != j)
                    {
                        continue;
                    }
                    sum[0] += config->image->content[k * 4];
                    sum[1] += config->image->content[k * 4 + 1];
                    sum[2] += config->image->content[k * 4 + 2];
                    count++;
                }
                if (count == 0)
                {
                    // use a sample from the center of the image
                    int index = size / 2;
                    centroids[j * 3] = config->image->content[index * 4];
                    centroids[j * 3 + 1] = config->image->content[index * 4 + 1];
                    centroids[j * 3 + 2] = config->image->content[index * 4 + 2];
                }
                else 
                {
                    centroids[j * 3] = sum[0] / count;
                    centroids[j * 3 + 1] = sum[1] / count;
                    centroids[j * 3 + 2] = sum[2] / count;
                }
            }
        }
#pragma omp for schedule(static, 1024)
        for (int i = 0; i < size; i++)
        {
            int index = cluster_indices[i];
            config->image->content[i * 4] = centroids[index * 3];
            config->image->content[i * 4 + 1] = centroids[index * 3 + 1];
            config->image->content[i * 4 + 2] = centroids[index * 3 + 2];
        }
    }

    free(centroids);
    free(cluster_indices);

    return omp_get_wtime() - start_time;
}

int distance(unsigned char *a, unsigned char *b, int a_index, int b_index)
{
    return sqrt(
        pow(a[a_index * 4] - b[b_index * 3], 2) + pow(a[a_index * 4 + 1] - b[b_index * 3 + 1], 2) + pow(a[a_index * 4 + 2] - b[b_index * 3 + 2], 2));
}

double compress_gpu(struct config * config)
{
	clock_t total_time;
	total_time = clock();

    int size = config->image->width * config->image->height;
    int pitch_size = config->image->pitch * config->image->height;

    unsigned char *centroids = (unsigned char *)malloc(config->cluster_count * 3 * sizeof(unsigned char));
    int *cluster_indices = (int *)malloc(size * sizeof(int));

    // initialize centroids
    // todo move to GPU kernel function
    initialize_centroids(config, centroids);

    cl_int status;

    char *source = (char *)malloc(MAX_SOURCE_SIZE);
    get_cl_source(source);

    cl_device_id device_id = get_device();

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
    cl_error_check(status, "Failed to create context");

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &status);
    cl_error_check(status, "Failed to create command queue");

    size_t local_item_size = WORKGROUP_SIZE;
    size_t num_groups = ((size - 1) / local_item_size + 1);
    size_t global_item_size = num_groups * local_item_size;

    cl_mem image_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pitch_size * sizeof(unsigned char), config->image->content, &status);
    cl_error_check(status, "Failed to create image buffer");

    cl_mem centroids_obj = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, config->cluster_count * 3 * sizeof(unsigned char), centroids, &status);
    cl_error_check(status, "Failed to create centroids buffer");

    cl_mem cluster_indices_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(int), NULL, &status);
    cl_error_check(status, "Failed to create cluster indices buffer");

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &status);
    cl_error_check(status, "Failed to create program");

    status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    size_t build_log_len;
    char *build_log;
    status |= clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
                                
    build_log = (char *)malloc(sizeof(char) * (build_log_len + 1));
    status |= clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);

    free(build_log);
    printf("%s\n", build_log);
    cl_error_check(status, "Failed to build program");

    cl_kernel kernel_associate_clusters = clCreateKernel(program, "associate_clusters", &status);

    status |= clSetKernelArg(kernel_associate_clusters, 0, sizeof(cl_mem), (void *)&image_mem_obj);
    status |= clSetKernelArg(kernel_associate_clusters, 1, sizeof(cl_mem), (void *)&centroids_obj);
    status |= clSetKernelArg(kernel_associate_clusters, 2, sizeof(cl_mem), (void *)&cluster_indices_obj);
    status |= clSetKernelArg(kernel_associate_clusters, 3, sizeof(cl_int), (void *)&size);
    status |= clSetKernelArg(kernel_associate_clusters, 4, sizeof(cl_int), (void *)&config->cluster_count);
    cl_error_check(status, "Failed to create associate_clusters kernel and set arguments");

    cl_kernel kernel_update_centroids = clCreateKernel(program, "update_centroids", &status);

    status |= clSetKernelArg(kernel_update_centroids, 0, sizeof(cl_mem), (void *)&image_mem_obj);
    status |= clSetKernelArg(kernel_update_centroids, 1, sizeof(cl_mem), (void *)&centroids_obj);
    status |= clSetKernelArg(kernel_update_centroids, 2, sizeof(cl_mem), (void *)&cluster_indices_obj);
    status |= clSetKernelArg(kernel_update_centroids, 3, sizeof(cl_int), (void *)&size);
    status |= clSetKernelArg(kernel_update_centroids, 4, sizeof(cl_int), (void *)&config->cluster_count);
    cl_error_check(status, "Failed to create update_centroids kernel and set arguments");

    for (int i = 0; i < config->iteration_count; i++) {
        status = clEnqueueNDRangeKernel(command_queue, kernel_associate_clusters, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        cl_error_check(status, "Failed to run associate_clusters");

        status = clEnqueueNDRangeKernel(command_queue, kernel_update_centroids, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        cl_error_check(status, "Failed to run update_centroids");
    }

    status = clEnqueueReadBuffer(command_queue, cluster_indices_obj, CL_TRUE, 0, size * sizeof(int), cluster_indices, 0, NULL, NULL);
    cl_error_check(status, "Failed to read cluster indices");

    status = clEnqueueReadBuffer(command_queue, centroids_obj, CL_TRUE, 0, config->cluster_count * 3 * sizeof(unsigned char), centroids, 0, NULL, NULL);
    cl_error_check(status, "Failed to read centroids");

    status = clFlush(command_queue);
    status |= clFinish(command_queue);
    status |= clReleaseKernel(kernel_associate_clusters);
    status |= clReleaseKernel(kernel_update_centroids);
    status |= clReleaseProgram(program);
    status |= clReleaseMemObject(image_mem_obj);
    status |= clReleaseMemObject(centroids_obj);
    status |= clReleaseMemObject(cluster_indices_obj);
    status |= clReleaseCommandQueue(command_queue);
    status |= clReleaseContext(context);
    cl_error_check(status, "Failed to clear memory");

    // update output TODO reuse input ???
    for (int i = 0; i < size; i++)
    {
        int index = cluster_indices[i];
        config->image->content[i * 4] = centroids[index * 3];
        config->image->content[i * 4 + 1] = centroids[index * 3 + 1];
        config->image->content[i * 4 + 2] = centroids[index * 3 + 2];
    }

    free(centroids);
    free(cluster_indices);

	total_time = clock() - total_time;
    return ((double)total_time) / CLOCKS_PER_SEC;
}

void get_cl_source(char* source)
{
    FILE *fp;
    size_t source_size;

    fp = fopen("kernelCompress.cl", "r");
    if (!fp)
    {
        error("Failed to open cl source");
    }
    source_size = fread(source, 1, MAX_SOURCE_SIZE, fp);
    source[source_size] = '\0';
    fclose(fp);
}

cl_device_id get_device()
{
    cl_int status;

    cl_platform_id platform_id[10];
    cl_uint ret_num_platforms;
    char *buf;
    size_t buf_len;
    status = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
    cl_error_check(status, "Failed to get platform IDs");

    cl_device_id device_id[10];
    cl_uint ret_num_devices;

    status = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10, device_id, &ret_num_devices);
    cl_error_check(status, "Failed to get device IDs");

    return device_id[0];
}
