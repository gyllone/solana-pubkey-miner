#ifndef VANITY_H
#define VANITY_H

#include <stddef.h>

#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -- Types ----------------------------------------------------------------- */

typedef struct {
	char name[256];
	size_t totalGlobalMem;
	int maxBlocksPerMultiProcessor;
	int maxThreadsPerBlock;
	int maxThreadsDim[3];
	int maxThreadsPerMultiProcessor;
} DeviceInfo;

typedef struct {
	bool found;
	char key[256];
	char pkey[256];
} Result;

typedef struct {
	bool enable_prefix;
	bool enable_suffix;
	char prefix[8];
	char suffix[8];
	int max_attempts_per_kernel;
	int max_attempts_per_task;
} Task;

/* -- Prototypes, Because C++ ----------------------------------------------- */

EXPORT int count_devices();
EXPORT DeviceInfo get_device_info(int device_id);
EXPORT Result vanity_run(int device_id, int blocks, int threads_per_block, Task task);

/* -- Entry Point ----------------------------------------------------------- */

#ifdef __cplusplus
}
#endif

#endif