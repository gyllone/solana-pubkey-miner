#include <chrono>

#include "curand_kernel.h"
#include "gpu_common.h"
#include "vanity.h"

#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"

// kernel functions declared here
NOT_EXPORTED __global__ void vanity_scan(Task* tasks, Result* results, unsigned long long seed);
NOT_EXPORTED __device__ bool b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz);

/* -- Entry Point ----------------------------------------------------------- */

int main(int argc, char const* argv[]) {
	// ed25519_set_verbose(true);

	// get devices
	int devices = count_devices();
	printf("Found %d devices\n\n", devices);

	// get device info
	for (int i = 0; i < devices; i++) {
		DeviceInfo info = get_device_info(i);
		printf("Device %d: %s\n", i, info.name);
		printf("  totalGlobalMem: %f MiB\n", (float)info.totalGlobalMem / 1024 / 1024);
		printf("  maxBlocksPerMultiProcessor: %d\n", info.maxBlocksPerMultiProcessor);
		printf("  maxThreadsPerBlock: %d\n", info.maxThreadsPerBlock);
		printf("  maxThreadsDim: %d, %d, %d\n", info.maxThreadsDim[0], info.maxThreadsDim[1], info.maxThreadsDim[2]);
		printf("  maxThreadsPerMultiProcessor: %d\n\n", info.maxThreadsPerMultiProcessor);
	}

	const int device_id = 0;
	const int BLOCKS = 16;
	const int THREADS_PER_BLOCK = 256;

	Task task;
	task.enable_prefix = true;
	task.enable_suffix = false;
	memcpy(task.prefix, "he??????", 8);
	memcpy(task.suffix, "????????", 8);
	task.max_attempts_per_kernel = 30000;
	task.max_attempts_per_task = 1000;

	Result result = vanity_run(device_id, BLOCKS, THREADS_PER_BLOCK, task);
	if (result.found) {
		printf("Vanity found! key: %s, pkey: %s\n", result.key, result.pkey);
	} else {
		printf("Vanity not found!\n");
	}
}

/* -- Vanity Step Functions ------------------------------------------------- */

int count_devices() {
	int count;
	CUDA_CHK(cudaGetDeviceCount(&count));
	return count;
}

DeviceInfo get_device_info(int device_id) {
	cudaDeviceProp prop;
	CUDA_CHK(cudaGetDeviceProperties(&prop, device_id));

	DeviceInfo info;
	memcpy(info.name, prop.name, 256);
	info.totalGlobalMem = prop.totalGlobalMem;
	info.maxBlocksPerMultiProcessor = prop.maxBlocksPerMultiProcessor;
	info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
	info.maxThreadsDim[0] = prop.maxThreadsDim[0];
	info.maxThreadsDim[1] = prop.maxThreadsDim[1];
	info.maxThreadsDim[2] = prop.maxThreadsDim[2];
	info.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
	return info;
}

Result vanity_run(int device_id, int blocks, int threads_per_block, Task task) {
	printf("Starting on device: %d\n", device_id);

	CUDA_CHK(cudaSetDevice(device_id));

	const int N = blocks * threads_per_block;
	printf("device: %d, blocks: %d, threads per block: %d\n", device_id, blocks, threads_per_block);

	// malloc task
	Task *dev_tasks, *tasks;
	tasks = (Task*)malloc(N * sizeof(Task));
	for (int i = 0; i < N; ++i) {
		tasks[i] = task;
	}
	CUDA_CHK(cudaMalloc((void **)&dev_tasks, N * sizeof(Task)));
	// copy task to device
	CUDA_CHK(cudaMemcpy(dev_tasks, tasks, N * sizeof(Task), cudaMemcpyHostToDevice));

	Result *dev_results, *results;
	results = (Result*)malloc(N * sizeof(Result));
	CUDA_CHK(cudaMalloc((void **)&dev_results, N * sizeof(Result)));

	// try to run scan kernel
	Result final_result = { false };
	for (int attempt = 0; attempt < task.max_attempts_per_task; attempt++) {
		auto start = std::chrono::high_resolution_clock::now();
		printf("Attempt: %d, running scan kernel... ", attempt+1);

		vanity_scan<<<blocks, threads_per_block>>>(dev_tasks, dev_results, time(NULL));
		CUDA_CHK(cudaDeviceSynchronize());

		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		printf("Scan kernel done, elapsed: %fs\n", elapsed.count());

		// copy result from device
		CUDA_CHK(cudaMemcpy(results, dev_results, N * sizeof(Result), cudaMemcpyDeviceToHost));
		// confirm result
		for (int i = 0; i < N; ++i) {
			if (results[i].found) {
				final_result = results[i];
				break;
			}
		}

		if (final_result.found) {
			break;
		}
	}

	// free device memory
	// CUDA_CHK(cudaFree(dev_states));
	CUDA_CHK(cudaFree(dev_tasks));
	CUDA_CHK(cudaFree(dev_results));
	// free host memory
	free(tasks);
	free(results);

	return final_result;
}

/* -- CUDA Vanity Functions ------------------------------------------------- */

// __global__ void vanity_init(curandState* states, unsigned long long seed) {
// 	int id = threadIdx.x + (blockIdx.x * blockDim.x);
// 	curandState state;
// 	curand_init(seed, id, 0, &state);
// 	states[id] = state;
// }

__global__ void vanity_scan(Task* tasks, Result* results, unsigned long long rng_seed) {
	int id = threadIdx.x + (blockIdx.x * blockDim.x);

	// Local Kernel State
	// curandState state = states[id];
	Task task = tasks[id];

	ge_p3 A;
	unsigned char seed[32]     = {0};
	unsigned char publick[32]  = {0};
	unsigned char privatek[64] = {0};
	char key[256]              = {0};
	char pkey[256]             = {0};

	// Start from an Initial Random Seed (Slow)
	// NOTE: Insecure random number generator, do not use keys generator by
	// this program in live.
	curandState state;
	curand_init(rng_seed, id, 0, &state);
	for (int i = 0; i < 8; i++) {
		unsigned int random = curand(&state);
		seed[i*4+0] = random >> 24;
		seed[i*4+1] = random >> 16;
		seed[i*4+2] = random >> 8;
		seed[i*4+3] = random;
	}
	// states[id] = state;

	// Generate Random Key Data
	sha512_context md;

	for (int attempt = 0; attempt < task.max_attempts_per_kernel; attempt++) {
		// sha512_init Inlined
		md.curlen   = 0;
		md.length   = 0;
		md.state[0] = UINT64_C(0x6a09e667f3bcc908);
		md.state[1] = UINT64_C(0xbb67ae8584caa73b);
		md.state[2] = UINT64_C(0x3c6ef372fe94f82b);
		md.state[3] = UINT64_C(0xa54ff53a5f1d36f1);
		md.state[4] = UINT64_C(0x510e527fade682d1);
		md.state[5] = UINT64_C(0x9b05688c2b3e6c1f);
		md.state[6] = UINT64_C(0x1f83d9abfb41bd6b);
		md.state[7] = UINT64_C(0x5be0cd19137e2179);

		// sha512_update inlined
		// 
		// All `if` statements from this function are eliminated if we
		// will only ever hash a 32 byte seed input. So inlining this
		// has a drastic speed improvement on GPUs.
		//
		// This means:
		//   * Normally we iterate for each 128 bytes of input, but we are always < 128. So no iteration.
		//   * We can eliminate a MIN(inlen, (128 - md.curlen)) comparison, specialize to 32, branch prediction improvement.
		//   * We can eliminate the in/inlen tracking as we will never subtract while under 128
		//   * As a result, the only thing update does is copy the bytes into the buffer.
		// const unsigned char *in = seed;
		// for (size_t i = 0; i < 32; i++) {
		// 	md.buf[i] = in[i];
		// }
		memcpy(md.buf, seed, 32);
		md.curlen = 32;

		// sha512_final inlined
		// 
		// As update was effectively elimiated, the only time we do
		// sha512_compress now is in the finalize function. We can also
		// optimize this:
		//
		// This means:
		//   * We don't need to care about the curlen > 112 check. Eliminating a branch.
		//   * We only need to run one round of sha512_compress, so we can inline it entirely as we don't need to unroll.
		md.length += md.curlen * UINT64_C(8);
		md.buf[md.curlen++] = 0x80;

		while (md.curlen < 120) {
			md.buf[md.curlen++] = 0;
		}

		STORE64H(md.length, md.buf+120);

		// Inline sha512_compress
		uint64_t S[8], W[80], t0, t1;

		/* Copy state into S */
		for (int i = 0; i < 8; i++) {
			S[i] = md.state[i];
		}

		/* Copy the state into 1024-bits into W[0..15] */
		for (int i = 0; i < 16; i++) {
			LOAD64H(W[i], md.buf + (8*i));
		}

		/* Fill W[16..79] */
		for (int i = 16; i < 80; i++) {
			W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];
		}

		/* Compress */
		#define RND(a,b,c,d,e,f,g,h,i) \
		t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i]; \
		t1 = Sigma0(a) + Maj(a, b, c);\
		d += t0; \
		h  = t0 + t1;

		for (int i = 0; i < 80; i += 8) {
			RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],i+0);
			RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],i+1);
			RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],i+2);
			RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],i+3);
			RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],i+4);
			RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],i+5);
			RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],i+6);
			RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],i+7);
		}

		#undef RND

		/* Feedback */
		for (int i = 0; i < 8; i++) {
			md.state[i] = md.state[i] + S[i];
		}

		// We can now output our finalized bytes into the output buffer.
		for (int i = 0; i < 8; i++) {
			STORE64H(md.state[i], privatek+(8*i));
		}

		// Code Until here runs at 87_000_000H/s.

		// ed25519 Hash Clamping
		privatek[0]  &= 248;
		privatek[31] &= 63;
		privatek[31] |= 64;

		// ed25519 curve multiplication to extract a public key.
		ge_scalarmult_base(&A, privatek);
		ge_p3_tobytes(publick, &A);

		// Code Until here runs at 87_000_000H/s still!

		size_t keysize = 256;
		b58enc(key, &keysize, publick, 32);

		// Code Until here runs at 22_000_000H/s. b58enc badly needs optimization.

		// We don't have access to strncmp/strlen here, I don't know
		// what the efficient way of doing this on a GPU is, so I'll
		// start with a dumb loop. There seem to be implementations out
		// there of bignunm division done in parallel as a CUDA kernel
		// so it might make sense to write a new parallel kernel to do
		// this.

		bool prefix_found = true;
		if (task.enable_prefix) {
			for (int i = 0; i < 8; i++) {
				char t = task.prefix[i];
				if (key[i] != t && t != '?') {
					prefix_found = false;
					break;
				}
			}
		}

		bool suffix_found = true;
		if (task.enable_suffix) {
			for (int i = 1; i <= 8; i++) {
				char t = task.suffix[8 - i];
				if (key[keysize - 1 - i] != t && t != '?') {
					suffix_found = false;
					break;
				}
			}
		}

		if (prefix_found && suffix_found) {
			size_t pkeysize = 256;
			b58enc(pkey, &pkeysize, seed, 32);

			Result result;
			result.found = true;
			memcpy(result.key, key, 256);
			memcpy(result.pkey, pkey, 256);

			results[id] = result;
			return;
		}

		// new seed = seed + 1
		for (int i = 0; i < 32; i++) {
			if (seed[i] == 255) {
				seed[i] = 0;
			} else {
				seed[i] += 1;
				break;
			}
		}
	}
}

__device__ bool b58enc(
	char    *b58,
    size_t  *b58sz,
    uint8_t *data,
    size_t  binsz
) {
	// Base58 Lookup Table
	const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

	const uint8_t *bin = data;
	int carry;
	size_t i, j, high, zcount = 0;
	size_t size;
	
	while (zcount < binsz && !bin[zcount])
		++zcount;
	
	size = (binsz - zcount) * 138 / 100 + 1;
	uint8_t buf[256];
	memset(buf, 0, size);
	
	for (i = zcount, high = size - 1; i < binsz; ++i, high = j)
	{
		for (carry = bin[i], j = size - 1; (j > high) || carry; --j)
		{
			carry += 256 * buf[j];
			buf[j] = carry % 58;
			carry /= 58;
			if (!j) {
				// Otherwise j wraps to maxint which is > high
				break;
			}
		}
	}
	
	for (j = 0; j < size && !buf[j]; ++j);
	
	if (*b58sz <= zcount + size - j) {
		*b58sz = zcount + size - j + 1;
		return false;
	}
	
	if (zcount) memset(b58, '1', zcount);
	for (i = zcount; j < size; ++i, ++j) b58[i] = b58digits_ordered[buf[j]];

	b58[i] = '\0';
	*b58sz = i + 1;
	
	return true;
}
