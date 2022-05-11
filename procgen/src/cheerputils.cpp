#include <cheerp/client.h>

// Offset might be 0, you can use it create a pointer to a specific element of the typed array
int *createData(client::Int32Array *a, int offset) {
    return __builtin_cheerp_make_regular<int>(a, offset);
}