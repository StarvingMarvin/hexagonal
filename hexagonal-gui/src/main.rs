use slint::{Image, SharedPixelBuffer, Rgba8Pixel};

fn low_level_render(_width: u32, _height: u32, _buffer: &mut [u8]) {
    // render beautiful circle or other shapes here
}

fn main() {
    MainWindow::new().unwrap().run().unwrap();


    let mut pixel_buffer = SharedPixelBuffer::<Rgba8Pixel>::new(320, 200);

    low_level_render(pixel_buffer.width(), pixel_buffer.height(),
                    pixel_buffer.make_mut_bytes());

    let _image = Image::from_rgba8(pixel_buffer);
}

slint::slint! {
    export component MainWindow inherits Window {
        Text {
            text: "hello world";
            color: green;
        }
    }
}
