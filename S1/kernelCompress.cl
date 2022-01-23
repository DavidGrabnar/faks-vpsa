inline int color_distance(unsigned char *a, unsigned char *b, int a_index, int b_index)
{
    int x = a[a_index * 4] - b[b_index * 3];
    int y = a[a_index * 4 + 1] - b[b_index * 3 + 1];
    int z = a[a_index * 4 + 2] - b[b_index * 3 + 2];
    return sqrt((double) (x * x + y * y + z * z));
}

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
    __global unsigned int *aggregates,
    int size,
    int centroid_count
    )
{
	int index = get_global_id(0);	
    
    if (index < centroid_count * 4)
    {
        aggregates[index] = 0;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (index < size)
    {
        int index_cluster = cluster_indices[index];
        atomic_add(&aggregates[index_cluster * 4], image_in[index * 4]);
        atomic_add(&aggregates[index_cluster * 4 + 1], image_in[index * 4 + 1]);
        atomic_add(&aggregates[index_cluster * 4 + 2], image_in[index * 4 + 2]);
        atomic_add(&aggregates[index_cluster * 4 + 3], 1);
    }
}

__kernel void update_values(
    __global unsigned char *image_in,
    __global unsigned char *centroids,
    __global unsigned int *aggregates,
    int size,
    int centroid_count
)
{
	int index = get_global_id(0);	

    if (index < centroid_count)
    {
        if (aggregates[index * 4 + 3] == 0)
        {
            // use a sample from the center of the image
            int index_center = size / 2;
            centroids[index * 3] = image_in[index_center * 4];
            centroids[index * 3 + 1] = image_in[index_center * 4 + 1];
            centroids[index * 3 + 2] = image_in[index_center * 4 + 2];
        }
        else 
        {
            centroids[index * 3] = aggregates[index * 4] / aggregates[index * 4 + 3];
            centroids[index * 3 + 1] = aggregates[index * 4 + 1] / aggregates[index * 4 + 3];
            centroids[index * 3 + 2] = aggregates[index * 4 + 2] / aggregates[index * 4 + 3];
        }
    }
}