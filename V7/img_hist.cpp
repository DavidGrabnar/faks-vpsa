#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "FreeImage.h"
#include <CL/cl.h>
#include <time.h>


#define BINS 256

#define WORKGROUP_SIZE (BINS)
#define MAX_SOURCE_SIZE 16384

#define DEFAULT_TARGET_GPU 1
#define DEFAULT_INPUT_PATH "640x480.jpg"

struct histogram
{
	unsigned int *R;
	unsigned int *G;
	unsigned int *B;
};

void histogramCPU(unsigned char *image_in, histogram H, int width, int height)
{
	printf("Running on CPU %d, %d\n", width, height);
	//Each color channel is 1 byte long, there are 4 channels BLUE, GREEN, RED and ALPHA
	//The order is BLUE|GREEN|RED|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
	for (int i = 0; i < (height); i++)
		for (int j = 0; j < (width); j++)
		{
			H.B[image_in[(i * width + j) * 4]]++;
			H.G[image_in[(i * width + j) * 4 + 1]]++;
			H.R[image_in[(i * width + j) * 4 + 2]]++;
		}
}

void histogramGPU(unsigned char *image_in, histogram H, int width, int height, int pitch)
{
	printf("Running on GPU %d, %d, %d\n", width, height, pitch);

	int pitch_image_size = pitch * height;
	int image_size = width * height;

	char ch;
	int i;
	cl_int ret;

	clock_t file_time;
	file_time = clock();

	// Branje datoteke
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("kernelHist.cl", "r");
	if (!fp)
	{
		fprintf(stderr, ":-(#\n");
		exit(1);
	}
	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
	fclose(fp);

	file_time = clock() - file_time;
	printf("Elapsed file time: %f seconds\n", ((double)file_time) / CLOCKS_PER_SEC);

	clock_t gpu_setup_time;
	gpu_setup_time = clock();

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

	printf("Sizes: %d, %d\n", local_item_size, global_item_size);
	clock_t copy_image_time;
	copy_image_time = clock();

	// Alokacija pomnilnika na napravi
	cl_mem image_in_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
											 pitch_image_size * sizeof(unsigned char), image_in, &ret);

	copy_image_time = clock() - copy_image_time;
	printf("Elapsed copy image time: %f seconds\n", ((double)copy_image_time) / CLOCKS_PER_SEC);

	cl_mem r_out_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
										  BINS * sizeof(unsigned int), NULL, &ret);

	cl_mem g_out_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
										  BINS * sizeof(unsigned int), NULL, &ret);

	cl_mem b_out_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
										  BINS * sizeof(unsigned int), NULL, &ret);

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
	cl_kernel kernel = clCreateKernel(program, "histogram_gpu", &ret);
	// program, ime "s"cepca, napaka

	// "s"cepec: argumenti
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_in_mem_obj);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&r_out_mem_obj);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&g_out_mem_obj);
	ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&b_out_mem_obj);
	ret |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&image_size);
	// "s"cepec, "stevilka argumenta, velikost podatkov, kazalec na podatke

	gpu_setup_time = clock() - gpu_setup_time;
	printf("Elapsed gpu setup time: %f seconds\n", ((double)gpu_setup_time) / CLOCKS_PER_SEC);

	clock_t program_run_time;
	program_run_time = clock();
	// "s"cepec: zagon
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
								 &global_item_size, &local_item_size, 0, NULL, NULL);
	// vrsta, "s"cepec, dimenzionalnost, mora biti NULL,
	// kazalec na "stevilo vseh niti, kazalec na lokalno "stevilo niti,
	// dogodki, ki se morajo zgoditi pred klicem

	program_run_time = clock() - program_run_time;
	printf("Elapsed program run time: %f seconds\n", ((double)program_run_time) / CLOCKS_PER_SEC);

	clock_t copy_result_time;
	copy_result_time = clock();
	// Kopiranje rezultatov
	ret = clEnqueueReadBuffer(command_queue, r_out_mem_obj, CL_TRUE, 0,
							  BINS * sizeof(unsigned int), H.R, 0, NULL, NULL);

	ret = clEnqueueReadBuffer(command_queue, g_out_mem_obj, CL_TRUE, 0,
							  BINS * sizeof(unsigned int), H.G, 0, NULL, NULL);

	ret = clEnqueueReadBuffer(command_queue, b_out_mem_obj, CL_TRUE, 0,
							  BINS * sizeof(unsigned int), H.B, 0, NULL, NULL);
	// branje v pomnilnik iz naparave, 0 = offset
	// zadnji trije - dogodki, ki se morajo zgoditi prej

	copy_result_time = clock() - copy_result_time;
	printf("Elapsed copy result time: %f seconds\n", ((double)copy_result_time) / CLOCKS_PER_SEC);

	clock_t clear_time;
	clear_time = clock();

	// "ci"s"cenje
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(image_in_mem_obj);
	ret = clReleaseMemObject(r_out_mem_obj);
	ret = clReleaseMemObject(g_out_mem_obj);
	ret = clReleaseMemObject(b_out_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	clear_time = clock() - clear_time;
	printf("Elapsed clear time: %f seconds\n", ((double)clear_time) / CLOCKS_PER_SEC);
}

void printHistogram(histogram H)
{
	printf("Colour\tNo. Pixels\n");
	for (int i = 0; i < BINS; i++)
	{
		if (H.B[i] > 0)
			printf("%dB\t%d\n", i, H.B[i]);
		if (H.G[i] > 0)
			printf("%dG\t%d\n", i, H.G[i]);
		if (H.R[i] > 0)
			printf("%dR\t%d\n", i, H.R[i]);
	}
}

int main(int argc, char *argv[])
{
	clock_t t;
	t = clock();

    //Initialize parameters
    int target_gpu = DEFAULT_TARGET_GPU;
    char *input_path = DEFAULT_INPUT_PATH;

    if (argc > 1 && !strcmp(argv[1], "0"))
    {
        target_gpu = 0;
    }
    if (argc > 2)
    {
        input_path = argv[2];
    }

	clock_t load_image_time;
	load_image_time = clock();

	//Load image from file
	FIBITMAP *imageBitmap = FreeImage_Load(FIF_JPEG, input_path, 0);
	//Convert it to a 32-bit image
	FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);

	//Get image dimensions
	int width = FreeImage_GetWidth(imageBitmap32);
	int height = FreeImage_GetHeight(imageBitmap32);
	int pitch = FreeImage_GetPitch(imageBitmap32);
	//Preapare room for a raw data copy of the image
	unsigned char *image_in = (unsigned char *)malloc(height * pitch * sizeof(unsigned char));

	//Initalize the histogram
	histogram H;
	H.B = (unsigned int *)calloc(BINS, sizeof(unsigned int));
	H.G = (unsigned int *)calloc(BINS, sizeof(unsigned int));
	H.R = (unsigned int *)calloc(BINS, sizeof(unsigned int));

	//Extract raw data from the image
	FreeImage_ConvertToRawBits(image_in, imageBitmap32, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

	//Free source image data
	FreeImage_Unload(imageBitmap32);
	FreeImage_Unload(imageBitmap);

	load_image_time = clock() - load_image_time;
	printf("Elapsed load image time: %f seconds\n", ((double)load_image_time)/CLOCKS_PER_SEC);

	clock_t historgram_run_time;
	historgram_run_time = clock();

	//Compute and print the histogram
	if (target_gpu) {
		histogramGPU(image_in, H, width, height, pitch);
	} else {
		histogramCPU(image_in, H, width, height);
	}
	
	historgram_run_time = clock() - historgram_run_time;
	printf("Elapsed histogram run time: %f seconds\n", ((double)historgram_run_time)/CLOCKS_PER_SEC);

	// printHistogram(H);

	t = clock() - t;
	printf("Elapsed total time: %f seconds\n", ((double)t)/CLOCKS_PER_SEC);

	return 0;
}

/*
Meritve:
------------

Čas ~ povprečje 5 meritev

 velikost (WxH) | Čas CPU (tc) [ms] | Čas GPU (tg) [ms] | Čas prenosa vhod [ms]	|  Pohitritev (S)
---------------------------------------------------------------------------------------------------
     640x480    |    	0,645     	|    	0,576     	|    	  0,832     	|      1,12    
     800x600    |    	1,023     	|     	0,739     	|    	  1,302     	|      1,38    
    1600x900    |    	2,991     	|     	1,369    	|    	  3,174     	|      4,05    
   1920x1080   	|    	4,333     	|     	1,676     	|    	  3,359     	|      2,59    
   3840x2160   	|      17,333     	|       5,058     	|    	 11,890     	|      3,43    

tc ~ čas obdelave z serijskim algoritmom na CPU
tg ~ čas obdelave s paralelnim algoritmom na GPU (brez časa prenosa vhodne slike, priprave programa itd.)

S = tc/tg ~ pohitritev

*/