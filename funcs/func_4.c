#include <stdio.h>
int main() {
    if (1 == 2) {
        printf("test");
    }
    int b = 0;
    switch(b) {
        case 1:
            printf("1");
        case 2:
            printf("2");
        default:
            printf("default");
    }
    while (b < 10) {
        b++;
    }
    return 0;
}