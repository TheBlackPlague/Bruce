#[cfg(target_os = "linux")]
#[test]
fn failed_tensorboard_writer_is_disabled_without_stopping_other_reporting()
{
    use super::{BufWriter, File, TensorBoard, tensorboard_action};

    let mut writer =
        Some(TensorBoard { writer: BufWriter::new(File::options().write(true).open("/dev/full").unwrap()) });
    writer.as_mut().unwrap().sample(1, 0.1, 0.001, 1.0).unwrap();
    tensorboard_action(&mut writer, TensorBoard::flush);
    assert!(writer.is_none());
    tensorboard_action(&mut writer, |writer| writer.sample(2, 0.1, 0.001, 1.0));
}
