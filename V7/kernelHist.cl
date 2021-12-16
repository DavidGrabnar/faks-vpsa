// kernel
__kernel void histogram_gpu(
    __global unsigned char *image_in,
    __global unsigned int *r_out,
    __global unsigned int *g_out,
    __global unsigned int *b_out,
    int width, 
    int height,
    int size
    )
{
    int i, j;

    // globalni indeks elementa							
	int index = get_global_id(0);	
	//izračun piksla
    i = index / width;
    j = index % width;

	//Each color channel is 1 byte long, there are 4 channels BLUE, GREEN, RED and ALPHA
	//The order is BLUE|GREEN|RED|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
    if (i < size)
    {
        b_out[image_in[(i * width + j) * 4]]++;
        g_out[image_in[(i * width + j) * 4 + 1]]++;
        r_out[image_in[(i * width + j) * 4 + 2]]++;
    }
}
