#include <stdio.h>
#include <stdlib.h>
#include "FreeImage.h"
#include <math.h>
#include <CL/cl.h>
#include <time.h>
#include <string.h>

#define WORKGROUP_SIZE	(512)
#define MAX_SOURCE_SIZE	16384

#define HEIGHT			(1024)
#define WIDTH			(1024)
#define OUTPUT_PATH		"mandelbrot.png"
#define TARGET_GPU		1

void mandelbrotCPU(unsigned char *image, int height, int width) {
	float x0, y0, x, y, xtemp;
	int i, j;
	int color;
	int iter;
	int max_iteration = 800;   //max stevilo iteracij
	unsigned char max = 255;   //max vrednost barvnega kanala

	//za vsak piksel v sliki							
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			x0 = (float)j / width * (float)3.5 - (float)2.5; //zacetna vrednost
			y0 = (float)i / height * (float)2.0 - (float)1.0;
			x = 0;
			y = 0;
			iter = 0;
			//ponavljamo, dokler ne izpolnemo enega izmed pogojev
			while ((x*x + y * y <= 4) && (iter < max_iteration))
			{
				xtemp = x * x - y * y + x0;
				y = 2 * x*y + y0;
				x = xtemp;
				iter++;
			}
			//izracunamo barvo (magic: http://linas.org/art-gallery/escape/smooth.html)
			color = 1.0 + iter - log(log(sqrt(x*x + y * y))) / log(2.0);
			color = (8 * max * color) / max_iteration;
			if (color > max)
				color = max;
			//zapisemo barvo RGBA (v resnici little endian BGRA)
			image[4 * i*width + 4 * j + 0] = 0; //Blue
			image[4 * i*width + 4 * j + 1] = color; // Green
			image[4 * i*width + 4 * j + 2] = 0; // Red
			image[4 * i*width + 4 * j + 3] = 255;   // Alpha
		}
}

void mandelbrotGPU(unsigned char *image, int height, int width) {
	int image_size = width * height;
	
	char ch;
	int i;
	cl_int ret;

	// Branje datoteke
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("kernelMandelbrot.cl", "r");
	if (!fp) 
	{
		fprintf(stderr, ":-(#\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
	fclose( fp );

	// Podatki o platformi
	cl_platform_id	platform_id[10];
	cl_uint			ret_num_platforms;
	char			*buf;
	size_t			buf_len;
	ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
			// max. "stevilo platform, kazalec na platforme, dejansko "stevilo platform
	
	// Podatki o napravi
	cl_device_id	device_id[10];
	cl_uint			ret_num_devices;
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
	size_t num_groups = ((image_size-1)/local_item_size+1);		
	size_t global_item_size = num_groups*local_item_size;		

	// Alokacija pomnilnika na napravi
	cl_mem image_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
									image_size * sizeof(unsigned char) * 4, NULL, &ret);

	// Priprava programa
	cl_program program = clCreateProgramWithSource(context,	1, (const char **)&source_str,  
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
	build_log =(char *)malloc(sizeof(char)*(build_log_len+1));
	ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 
								build_log_len, build_log, NULL);
	printf("%s\n", build_log);
	free(build_log);

	// "s"cepec: priprava objekta
	cl_kernel kernel = clCreateKernel(program, "mandelbrot_gpu", &ret);
			// program, ime "s"cepca, napaka


	// "s"cepec: argumenti
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_mem_obj);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&height);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&width);
	ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&image_size);
			// "s"cepec, "stevilka argumenta, velikost podatkov, kazalec na podatke

	// "s"cepec: zagon
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,						
								&global_item_size, &local_item_size, 0, NULL, NULL);	
			// vrsta, "s"cepec, dimenzionalnost, mora biti NULL, 
			// kazalec na "stevilo vseh niti, kazalec na lokalno "stevilo niti, 
			// dogodki, ki se morajo zgoditi pred klicem
																						
	// Kopiranje rezultatov
	ret = clEnqueueReadBuffer(command_queue, image_mem_obj, CL_TRUE, 0,						
							image_size * sizeof(unsigned char) * 4, image, 0, NULL, NULL);				
			// branje v pomnilnik iz naparave, 0 = offset
			// zadnji trije - dogodki, ki se morajo zgoditi prej
			
	// "ci"s"cenje
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(image_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
}

int main(int argc, char **argv)
{
	clock_t t;
	t = clock();

	// Parameters
	int height = WIDTH;
	int width = HEIGHT;
	char* output_path = OUTPUT_PATH;
	int target_gpu = 1;

	if (argc > 1) {
		width = atoi(argv[1]);
	}
	if (argc > 2) {
		height = atoi(argv[2]);
	}
	if (argc > 3) {
		output_path = argv[3];
	}
	if (argc > 4 && !strcmp(argv[4], "0")) {
		target_gpu = 0;
	}

	int pitch = ((32 * width + 31) / 32) * 4;
	int image_size = width * height;

	// Rezervacija pomnilnika
	//rezerviramo prostor za sliko (RGBA)
	unsigned char *image = (unsigned char *)malloc(image_size * sizeof(unsigned char) * 4);

	if (target_gpu) {
		printf("Using GPU\n");
		mandelbrotGPU(image, height, width);
	} else {
		printf("Using CPU\n");
		mandelbrotCPU(image, height, width);
	}
    
    // Prikaz rezultatov
	FIBITMAP *dst = FreeImage_ConvertFromRawBits(image, width, height, pitch,
		32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
	FreeImage_Save(FIF_PNG, dst, output_path, 0);

    free(image);

	t = clock() - t;
	double time_taken = ((double)t)/CLOCKS_PER_SEC; // calculate the elapsed time
	printf("Elapsed time: %f seconds\n", time_taken);

	return 0;
}

/*

Meritve:
------------

Čas ~ povprečje 5 meritev

 velikost (WxH) |  Čas CPU (tc) [s] |  Čas GPU (tg) [s]	| Pohitritev (S)
-------------------------------------------------------------------------
     640x480    |    	0,310     	|    	0,196     	|     1,58      
     800x600    |    	0,482     	|     	0,212     	|	  2,27
    1600x900    |    	1,427     	|     	0,270    	|	  5,29
   1920x1080   	|    	2,050     	|     	0,311     	|	  6,59
   3840x2160   	|    	8,154     	|       0,704     	|	 11,58

tc ~ čas z serijskim algoritmom na CPU
tg ~ čas s paralelnim algoritmom na GPU

S = tc/tg ~ pohitritev

Ugotovitve:
------------

Upočasnitev serijskega algoritma na CPU je linearna z številom pikslov (= WxH).
V spodnji tabeli je primerjava povečave števila pisklov in povečava časa serijskega algoritma z prejšnjo meritvijo, kjer je to razvidno.

Za razliko serijskega algoritma, se paralelni algoritem na GPU povečuje veliko počasneje.

#velikost (WxH) |  #pikslov | Povečava #pikslov | Čas CPU (tg) [s]	| Povečava časa CPU |  Čas GPU (tg) [s]	| Povečava časa GPU
--------------------------------------------------------------------------------------------------------------------------------
     640x480    |   307200  |    	  /     	|     	0,310      	|		  /	     	|    	0,196		|		 /
     800x600    |   480000  |  		1,563    	|	  	0,482		|	  	1,555     	|     	0,212		|		1,08
    1600x900    |  1440000  |  		3,000   	|	  	1,427		|	  	2,961     	|     	0,270		|		1,27
   1920x1080   	|  2073600  |  		1,440    	|	  	2,050		|	  	1,437     	|     	0,311		|		1,15
   3840x2160   	|  8294400  |  		4,000     	|	  	8,154		|	  	3,978     	|       0,704  		|		2,26

*/
