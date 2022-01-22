inline int color_distance(unsigned char *a, unsigned char *b, int a_index, int b_index)
{
    int x = a[a_index * 4] - b[b_index * 3];
    int y = a[a_index * 4 + 1] - b[b_index * 3 + 1];
    int z = a[a_index * 4 + 2] - b[b_index * 3 + 2];
    return sqrt((double) (x * x + y * y + z * z));
}

// kernel
__kernel void associate_clusters(
    __global unsigned char *image_in,
    __global unsigned char *centroids,
    __global int *cluster_indices,
    int size,
    int centroid_count
    )
{
	int index = get_global_id(0);	

    if (index < size)
    {
        int min_index = 0;
        int min_distance = color_distance(image_in, centroids, index, 0);
        for (int k = 1; k < centroid_count; k++)
        {
            int d = color_distance(image_in, centroids, index, k);
            if (d < min_distance)
            {
                min_index = k;
                min_distance = d;
            }
        }
        cluster_indices[index] = min_index;
    }
}

__kernel void update_centroids(
    __global unsigned char *image_in,
    __global unsigned char *centroids,
    __global int *cluster_indices,
    int size,
    int centroid_count
    )
{
	int index = get_global_id(0);	
    
    if (index < centroid_count)
    {
        long sum[3] = {0, 0, 0};
        int count = 0;
        for (int k = 0; k < size; k++)
        {
            if (cluster_indices[k] != index)
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
            // use a sample from the center of the image
            int center_index = size / 2;
            centroids[index * 3] = image_in[center_index * 4];
            centroids[index * 3 + 1] = image_in[center_index * 4 + 1];
            centroids[index * 3 + 2] = image_in[center_index * 4 + 2];
        }
        else
        {
            centroids[index * 3] = sum[0] / count;
            centroids[index * 3 + 1] = sum[1] / count;
            centroids[index * 3 + 2] = sum[2] / count;
        }
    }
}
