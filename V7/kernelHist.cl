// kernel
__kernel void histogram_gpu(
    __global unsigned char *image_in,
    __global unsigned int *r_out,
    __global unsigned int *g_out,
    __global unsigned int *b_out,
    int size
    )
{
    __local unsigned int r_local[256];
    __local unsigned int g_local[256];
    __local unsigned int b_local[256];

    int local_index = get_local_id(0);
	int global_index = get_global_id(0);

    r_local[local_index] = 0;
    g_local[local_index] = 0;
    b_local[local_index] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

	//Each color channel is 1 byte long, there are 4 channels BLUE, GREEN, RED and ALPHA
	//The order is BLUE|GREEN|RED|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
    if (global_index < size)
    {
        atomic_inc(&b_local[image_in[global_index * 4]]);
        atomic_inc(&g_local[image_in[global_index * 4 + 1]]);
        atomic_inc(&r_local[image_in[global_index * 4 + 2]]);
    
        barrier(CLK_LOCAL_MEM_FENCE);

        atomic_add(&b_out[local_index], b_local[local_index]);
        atomic_add(&g_out[local_index], g_local[local_index]);
        atomic_add(&r_out[local_index], r_local[local_index]);
    }
}
