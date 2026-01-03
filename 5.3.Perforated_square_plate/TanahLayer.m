classdef TanahLayer < nnet.layer.Layer
    properties (Learnable)
        Alpha % 可学习参数
    end
    properties (Hidden)
        AlphaHistory;  % 记录 Alpha 的历史值（单元数组）
    end
    methods
        function layer = TanahLayer(numChannels, name)
            % 调用父类构造函数（可选，但建议传递名称）
            layer = layer@nnet.layer.Layer();

            % 设置层名称
            if nargin < 2
                layer.Name = 'tanah';
            else
                layer.Name = name;
            end

            % 初始化可学习参数Alpha（维度需与输入通道匹配）
            layer.Alpha = ones(numChannels,1);  % 初始化为列向量
            layer.AlphaHistory = {};            % 初始化历史记录
        end

        function Z = predict(layer, X)
            % 前向传播计算
            Z = tanh(layer.Alpha.*X);
        end
    end
end

