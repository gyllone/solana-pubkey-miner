#ifndef EXPORT_LIB_H
#define EXPORT_LIB_H

#define EXPORT __attribute__((visibility("default")))
#define NOT_EXPORTED __attribute__((visibility("hidden")))

#endif