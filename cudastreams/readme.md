### usage
```c++

int num_streams = 1;
cudaStream_t *streams = (cudaStream_t *) malloc(num_streams * sizeof(cudaStream_t));
for (int i = 0; i < num_streams; i++) 	cudaStreamCreate(&(streams[i]));

for(int sid=0; sid < num_streams; sid++)
{

}

for (int i = 0; i < num_streams; i++) 	cudaStreamDestroy(streams[i]);

```
