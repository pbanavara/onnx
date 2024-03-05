//
//  Utils.swift
//  onnx
//
//  Created by Pradeep Banavara on 05/03/24.
//

import Foundation
import SwiftUI
import onnxruntime_objc

func convertImageToTensor() -> NSMutableData{
    
    let image = UIImage(named: "IMG_0688")?.resized(to: CGSize(width: 640.0, height: 640.0))
    let data = image?.pngData()
    let count = data?.count
    let rawData = NSMutableData(data: data!)
    print(image?.size)
    do {
        guard let modelPath = Bundle.main.path(forResource: "yolov8n-pose", ofType: "onnx") else {
            fatalError("Model file not found")
        }
        
        let ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.error)
        let ortSession = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: nil)
        let input = try ORTValue(tensorData:rawData, elementType:ORTTensorElementDataType.float, shape:[])
        //try ortSession.run(withInputs: ["input": tensorValue], outputs: ["output": opTensorValue], runOptions:  ORTRunOptions())
        let outputs = try ortSession.run(
                        withInputs: ["input": input],
                        outputNames: ["output"],
                        runOptions: nil)
        guard let output = outputs["output"] else {
            fatalError("Failed to get model output from inference.")
        }
        
        return try output.tensorData()
    } catch {
        print(error)
        fatalError("Error in running the ONNX model")
    }
}

extension UIImage {
  func resized(to newSize: CGSize) -> UIImage? {
    UIGraphicsBeginImageContextWithOptions(newSize, false, 0)
    defer { UIGraphicsEndImageContext() }

    draw(in: CGRect(origin: .zero, size: newSize))
    return UIGraphicsGetImageFromCurrentImageContext()
  }
}
