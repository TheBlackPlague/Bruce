#[derive(Clone, PartialEq, prost::Message)]
pub struct Event {
    #[prost(double, tag = "1")]
    pub wall_time: f64,
    #[prost(int64, tag = "2")]
    pub step: i64,
    #[prost(string, optional, tag = "3")]
    pub file_version: Option<String>,
    #[prost(message, optional, tag = "5")]
    pub summary: Option<Summary>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Summary {
    #[prost(message, repeated, tag = "1")]
    pub value: Vec<Value>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Value {
    #[prost(string, tag = "1")]
    pub tag: String,
    #[prost(float, tag = "2")]
    pub simple_value: f32,
}
