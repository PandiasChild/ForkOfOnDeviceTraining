#ifndef NPYLOADER_H
#define NPYLOADER_H

FILE* openNPYFile(char* path);

void checkMagic(FILE* f);

uint32_t readHeaderSize(FILE* f);

char* readHeader(char* header, uint32_t headerSize, FILE* f);

dtype_t parseHeaderDescription(char* s);

dtype_t getDTypeFromHeader(char* header);

size_t getNumberOfDimsFromHeader(const char* header);

void getShapeFromHeader(shape_t* shape, size_t* dims, size_t* orderOfDims, char* header,
                            size_t numberOfDims);

#endif //NPYLOADER_H
