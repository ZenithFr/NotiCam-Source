#include <android/neuralnetworks.h>
#include <cstdio>
#include <vector>
#include <string>

int main() {
    uint32_t numDevices = 0;
    ANeuralNetworks_getDeviceCount(&numDevices);
    printf("NNAPI Devices Count: %u\n", numDevices);

    ANeuralNetworksDevice* neuronDevice = nullptr;
    ANeuralNetworksDevice* mdlaDevice = nullptr;

    for (uint32_t i = 0; i < numDevices; i++) {
        ANeuralNetworksDevice* dev = nullptr;
        ANeuralNetworks_getDevice(i, &dev);
        const char* name = nullptr;
        int32_t type = 0;
        ANeuralNetworksDevice_getName(dev, &name);
        ANeuralNetworksDevice_getType(dev, &type);
        printf("Device [%u]: %s (type=%d)\n", i, name ? name : "null", type);
        if (name && std::string(name) == "mtk-neuron_shim") neuronDevice = dev;
        if (name && std::string(name) == "mtk-mdla_shim") mdlaDevice = dev;
    }

    if (!neuronDevice) {
        printf("ERROR: mtk-neuron_shim not found!\n");
        return 1;
    }

    // Helper lambda to test an op on mtk-neuron_shim
    auto testOp = [&](const char* opName, int32_t opType, int32_t operandType) {
        ANeuralNetworksModel* model = nullptr;
        ANeuralNetworksModel_create(&model);

        // Operand 0: Input tensor [1, 16, 16, 32]
        uint32_t inDims[] = {1, 16, 16, 32};
        ANeuralNetworksOperandType inType = {
            .type = operandType,
            .dimensionCount = 4,
            .dimensions = inDims,
            .scale = 0.0f,
            .zeroPoint = 0
        };
        ANeuralNetworksModel_addOperand(model, &inType);

        // For Conv2D / Depthwise / Add / Resize / Pad
        // We will build minimal valid graphs for each
        ANeuralNetworksModel_free(model);
    };

    printf("Neuron device pointer: %p\n", neuronDevice);
    return 0;
}
