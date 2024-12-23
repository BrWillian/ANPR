package main

/*
#cgo LDFLAGS: -L. -lANPR
#include "../meta/c_wrapper.h"
*/
import "C"
import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"time"
	"unsafe"
)

type ANPR_DETECT struct {
	ptr *C.anpr_t
}

func OCR_GetVersion() string {
	version := C.GoString(C.C_ANPRVERSION())

	return version
}

func OCR_Create() ANPR_DETECT {
	var anpr ANPR_DETECT
	anpr.ptr = C.C_ANPRCREATE()

	return anpr
}
func OCR_Delete(anpr ANPR_DETECT) {
	C.C_ANPRDELETE(anpr.ptr)
}

func OCR_Inference(anpr ANPR_DETECT, img image.Image) string {
	buf := new(bytes.Buffer)
	_ = jpeg.Encode(buf, img, nil)

	b := buf.Bytes()

	result := C.C_ANPRINFERENCE(anpr.ptr, (*C.uchar)(unsafe.Pointer(&b[0])), C.int(buf.Len()))

	return C.GoString(result)
}
func GetImageFromFilePath(filePath string) (image.Image, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	image, _, err := image.Decode(f)
	return image, err
}

func main() {
	pointer := OCR_Create()
	fmt.Println(pointer)

	files, err := ioutil.ReadDir("../images/")
	if err != nil {
		log.Fatal(err)
	}

	for _, file := range files {

		img, _ := GetImageFromFilePath("../images/" + file.Name())

		fmt.Println(file.Name())

		start := time.Now()

		result := OCR_Inference(pointer, img)

		duration := time.Since(start)

		fmt.Println(result)

		fmt.Printf("Time: %d ms\n", duration.Milliseconds())
	}

	fmt.Println("Destruct Pointer")

	OCR_Delete(pointer)

	fmt.Println("Destroyed Pointer")

	fmt.Println(OCR_GetVersion())
}

