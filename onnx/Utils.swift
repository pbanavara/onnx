//
//  Utils.swift
//  onnx
//
//  Created by Pradeep Banavara on 05/03/24.
//

import Foundation
import SwiftUI
import onnxruntime_objc

func plotPose() -> CGRect{
    do {
        guard let modelPath = Bundle.main.path(forResource: "yolov8n-pose-pre", ofType: "onnx") else {
            fatalError("Model file not found")
        }
        guard
            let input_image_url = Bundle.main.url(forResource: "IMG_0688", withExtension: "jpeg")
        else {
            fatalError("Failed to get image URL")
        }
        let inputData = try Data(contentsOf: input_image_url)
        let inputDataLength = inputData.count
        let inputShape = [NSNumber(integerLiteral: inputDataLength)]
        let inputTensor = try ORTValue(tensorData: NSMutableData(data: inputData), elementType:ORTTensorElementDataType.uInt8, shape:inputShape)

        let ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.info)
        let ortSessionOptions = try ORTSessionOptions()
        try ortSessionOptions.registerCustomOps(functionPointer: RegisterCustomOps)
        let ortSession = try ORTSession(
            env: ortEnv, modelPath: modelPath, sessionOptions: ortSessionOptions)
        let inputNames = try ortSession.inputNames()
        let outputNames = try ortSession.outputNames()
        let outputs = try ortSession.run(
            withInputs: [inputNames[0]: inputTensor], outputNames: Set(outputNames), runOptions: nil)
        
        // If the image is specified as the output layer in the model
        /*
        guard let outputTensor = outputs[outputNames[1]] else {
            fatalError("Failed to get model output from inference.")
        }
         */
        guard let outputTensor = outputs[outputNames[0]] else {
            fatalError("Failed to get model keypoint output from inference.")
        }
        
        let rect = try convertOutputTensorToImage(opTensor: outputTensor, inputImageData: inputData)
        //try convertOutputTensorToImage(opTensor: dataTensor)
        return rect
        
        
    } catch {
        print(error)
        fatalError("Error in running the ONNX model")
    }
}

func convertOutputTensorToImage(opTensor: ORTValue, inputImageData: Data) throws -> CGRect{
    let outputTypeAndShape = try opTensor.tensorTypeAndShapeInfo()
    let skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    
    let output = try opTensor.tensorData() as Data
    var arr2 = Array<UInt8>(repeating: 0, count: output.count/MemoryLayout<UInt8>.stride)
    _ = arr2.withUnsafeMutableBytes { output.copyBytes(to: $0) }
    
    print(arr2)
    print(arr2.count)
    
    let img = UIImage(data: inputImageData)
    let width = img?.size.width
    let height = img?.size.height
    print(width, height)
    let box = arr2[0..<4]
    let x = Double(Double(box[0]))
    let y = Double(Double(box[1]))
    let w = Double(box[2] - box[0])
    let h = Double(box[3] - box[1])
    
    return CGRect(x: x, y: y, width: w, height:h)
    
}

extension UIImage {
    func resized(to newSize: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(newSize, false, 0)
        defer { UIGraphicsEndImageContext() }

        draw(in: CGRect(origin: .zero, size: newSize))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
 }

