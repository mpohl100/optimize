use crate::neural::nn::shape::LayerShape;
use crate::neural::nn::shape::NeuralNetworkShape;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayerChangeType {
    Add,
    Remove,
    Change,
    None,
}

pub struct AnnotatedLayerShape {
    pub layer: LayerShape,
    pub change_type: LayerChangeType,
}

impl AnnotatedLayerShape {
    #[must_use]
    pub const fn new(
        layer: LayerShape,
        change_type: LayerChangeType,
    ) -> Self {
        Self { layer, change_type }
    }
}

pub struct AnnotatedNeuralNetworkShape {
    pub layers: Vec<AnnotatedLayerShape>,
}

impl AnnotatedNeuralNetworkShape {
    #[must_use]
    pub fn new(layers: NeuralNetworkShape) -> Self {
        let annotated_layers = layers
            .layers
            .iter()
            .map(|layer| AnnotatedLayerShape::new(layer.clone(), LayerChangeType::None))
            .collect();
        Self { layers: annotated_layers }
    }

    pub fn add_layer(
        &mut self,
        position: usize,
        layer: LayerShape,
    ) {
        self.layers.insert(position, AnnotatedLayerShape::new(layer, LayerChangeType::Add));
    }

    pub fn remove_layer(
        &mut self,
        position: usize,
    ) {
        self.layers.remove(position);
    }

    #[must_use]
    pub fn get_layer(
        &self,
        position: usize,
    ) -> &LayerShape {
        &self.layers[position].layer
    }

    #[must_use]
    pub fn get_annotated_layer(
        &self,
        position: usize,
    ) -> &AnnotatedLayerShape {
        &self.layers[position]
    }

    pub fn change_layer(
        &mut self,
        position: usize,
        layer: LayerShape,
    ) {
        self.layers[position] = AnnotatedLayerShape::new(layer, LayerChangeType::Change);
    }

    #[must_use]
    pub fn to_neural_network_shape(&self) -> NeuralNetworkShape {
        let layers =
            self.layers.iter().map(|annotated_layer| annotated_layer.layer.clone()).collect();
        NeuralNetworkShape::new(layers)
    }
}
