### inline smid and clock and return function value

```c++
#define DEVICE_INTRINSIC_QUALIFIERS   __device__ __forceinline__

DEVICE_INTRINSIC_QUALIFIERS
unsigned int smid()
{
	unsigned int r;
	asm("mov.u32 %0, %%smid;" : "=r"(r));
	return r;
}

DEVICE_INTRINSIC_QUALIFIERS
unsigned int timer()
{
	unsigned int r;
	asm("mov.u32 %0, %%clock;" : "=r"(r));
	return r;
}
```
