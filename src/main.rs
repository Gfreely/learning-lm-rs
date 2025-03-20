mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;


use crate::kvcache::KVCache;

use std::path::PathBuf;
use tokenizers::Tokenizer;
use iced::{button, scrollable, text_input, Alignment, Application, Button, Column, Command,
           Container, Element, Length, Row, Sandbox, Scrollable, Settings, Text, TextInput,container};
use iced::alignment::{Horizontal, Vertical};
use iced::Color;
use serde::{Deserialize, Serialize};

pub struct GenerationParams {
    pub max_turns: usize,
    pub max_len: usize,
    pub top_p: f32,
    pub top_k: u32,
    pub temperature: f32,
}

impl GenerationParams {
    pub fn new(max_turns: usize, max_len: usize, top_p: f32, top_k: u32, temperature: f32) -> Self {
        Self {
            max_turns,
            max_len,
            top_p,
            top_k,
            temperature,
        }
    }

    pub fn default() -> Self {
        Self {
            max_turns:10,
            max_len: 300,
            top_p: 0.8,
            top_k: 30,
            temperature: 1.,
        }
    }

    pub fn set_params(
        &mut self,
        max_turns: Option<usize>,
        max_len: Option<usize>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<(), &'static str> {
        if let Some(v) = max_turns {
            if v <= 0 {
                return Err("max_turns 必须大于0");
            }
            self.max_len = v;
        }
        if let Some(v) = max_len {
            if v <= 0 {
                return Err("max_len 必须大于0");
            }
            self.max_len = v;
        }

        if let Some(v) = top_p {
            if !(0.0..=1.0).contains(&v) {
                return Err("top_p 必须在 0.0 到 1.0 之间");
            }
            self.top_p = v;
        }

        if let Some(v) = top_k {
            if v == 0 {
                return Err("top_k 必须大于0");
            }
            self.top_k = v;
        }

        if let Some(v) = temperature {
            if v <= 0.0 {
                return Err("temperature 必须大于0");
            }
            self.temperature = v;
        }

        Ok(())
    }
}


const SYSTEM_PROMPT: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
const USER_PREFIX: &str = "<|im_start|>user\n";
const ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n";
const END_MARKER: &str = "<|im_end|>\n";


fn story_mode(input: &str)-> String{
    let mut params = GenerationParams::default();
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);
    let output_ids = llama.generate_story(
        input_ids,
        params.max_len,
        params.top_p,
        params.top_k,
        params.temperature,
    );
    let response=tokenizer.decode(&output_ids, true).unwrap();
    response.replace("<|end_story|>", "").trim().to_string()
}


fn chat_mode(user_input: &str,mut cache: &mut KVCache<f32>) -> String {
    let mut params = GenerationParams::default();
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut input = String::new();
    input = format!("{}{}{}{}",USER_PREFIX,user_input,END_MARKER,ASSISTANT_PREFIX) ;
    // 编码输入
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    // 生成回复
    let result = llama.generate_chat(
        input_ids,
        params.max_len,  // 每轮最大生成长度
        params.top_p,
        params.top_k,
        params.temperature,
        &mut cache
    );
    let response=tokenizer.decode(&result, true).unwrap();
    response.replace(END_MARKER, "").trim().to_string()
}

/*
fn chat_mode(){
    let mut params = GenerationParams::default();
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();


    let mut cache = llama.new_cache();

    // 对话模板常量
    const SYSTEM_PROMPT: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    const USER_PREFIX: &str = "<|im_start|>user\n";
    const ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n";
    const END_MARKER: &str = "<|im_end|>\n";


    for turn in 0..params.max_turns {
        // 获取用户输入
        let mut user_input = String::new();
        println!("\nTurn {}\nUser:", turn + 1);
        std::io::stdin().read_line(&mut user_input).unwrap();
        user_input = user_input.trim().to_string();

        if user_input.to_lowercase() == "exit" {
            break;
        }
        let mut input = String::new();
        input = format!("{}{}{}{}",USER_PREFIX,user_input,END_MARKER,ASSISTANT_PREFIX) ;
        //println!("{}",input);
        // 编码输入
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        //println!("Assistant:");

        // 生成回复
        let result = llama.generate_chat(
            input_ids,
            params.max_len,  // 每轮最大生成长度
            params.top_p,
            params.top_k,
            params.temperature,
            &mut cache
        );

        // 解码并更新历史
        let response = tokenizer.decode(&result, true).unwrap();
        let clean_response = response.replace(END_MARKER, "").trim().to_string();
        println!("Assistant: {}", clean_response);
    }

}

fn main(){
    chat_mode();
    //let response = story_mode("Once upon a time，");
    //println!("{}",response);
}
*/

#[derive(Debug, Clone, PartialEq)]
enum Mode {
    Chat,
    Story,
}

#[derive(Debug, Clone)]
enum Message {
    InputChanged(String),
    SendMessage,
    SwitchMode(Mode),
}

struct ChatApp{
    // 模式状态
    current_mode: Mode,

    // 输入状态
    chat_input: text_input::State,
    story_input: text_input::State,
    input_value: String,

    // 消息记录
    chat_messages: Vec<String>,
    story_messages: Vec<String>,

    // 按钮状态
    send_button: button::State,
    mode_buttons: (button::State, button::State),

    // 滚动状态
    scroll: scrollable::State,
    project_dir: String,
    model_dir: PathBuf,
    llama: model::Llama<f32>,
    cache:KVCache<f32>,
    counter: i16,

}

impl Application for ChatApp {
    type Executor = iced::executor::Default;
    type Message = Message;
    type Flags = ();

    fn new( _flags: ()) -> (Self, Command<Self::Message>) {
        let project_dir = env!("CARGO_MANIFEST_DIR").to_string();
        let model_dir = PathBuf::from(&project_dir).join("models").join("chat");
        let llama = model::Llama::<f32>::from_safetensors(&model_dir);
        let mut cache =llama.new_cache();
        (
            Self {
                current_mode: Mode::Chat,
                chat_input: text_input::State::new(),
                story_input: text_input::State::new(),
                input_value: String::new(),
                chat_messages: Vec::new(),
                story_messages: Vec::new(),
                send_button: button::State::new(),
                mode_buttons: (button::State::new(), button::State::new()),
                scroll: scrollable::State::new(),
                project_dir,
                model_dir,
                llama,
                cache,
                counter: 0,
                },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("Multi-Mode Iced Chat")
    }

    fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
        match message {
            Message::InputChanged(value) => {
                self.input_value = value;
            }
            Message::SendMessage => {
                if !self.input_value.is_empty() {

                    let response = match self.current_mode {
                        Mode::Chat => chat_mode(&self.input_value, &mut self.cache),
                        Mode::Story => story_mode(&self.input_value),
                    };

                    let messages = match self.current_mode {
                        Mode::Chat => &mut self.chat_messages,
                        Mode::Story => &mut self.story_messages,
                    };

                    messages.push(format!("You: {}", self.input_value));
                    messages.push(format!("System: {}", response));
                    self.input_value.clear();
                }
            }
            Message::SwitchMode(mode) => {
                self.current_mode = mode;
                // 切换时清空当前输入
                self.input_value.clear();
            }
        }
        Command::none()
    }

    fn view(&mut self) -> Element<Self::Message> {
        // 侧边栏
        let sidebar = Column::new()
            .padding(20)
            .spacing(20)
            .width(Length::Units(150))
            .push(mode_button(
                &mut self.mode_buttons.0,
                "Chat Mode",
                Mode::Chat,
                &self.current_mode,
            ))
            .push(mode_button(
                &mut self.mode_buttons.1,
                "Story Mode",
                Mode::Story,
                &self.current_mode,
            ));

        // 消息显示区域
        let messages = match self.current_mode {
            Mode::Chat => &self.chat_messages,
            Mode::Story => &self.story_messages,
        };

        let message_list = Scrollable::new(&mut self.scroll)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(20)
            .spacing(10)
            .push(
                Column::with_children(
                    messages.iter().map(|msg| {
                        Container::new(
                            Text::new(msg)
                                .size(16)
                                .color(if msg.starts_with("You:") {
                                    Color::from_rgb(0.1, 0.1, 0.5)
                                } else {
                                    Color::from_rgb(0.2, 0.5, 0.2)
                                })
                        )
                            .padding(10)
                            .style(MessageBubbleStyle)
                            .into()
                    }).collect()
                )
            );

        // 输入区域
        let input_area = Row::new()
            .spacing(10)
            .padding(10)
            .align_items(Alignment::Center)
            .push(
                TextInput::new(
                    match self.current_mode {
                        Mode::Chat => &mut self.chat_input,
                        Mode::Story => &mut self.story_input,
                    },
                    "Type your message...",
                    &self.input_value,
                    Message::InputChanged,
                )
                    .padding(10)
                    .on_submit(Message::SendMessage)
                    .width(Length::Fill),
            )
            .push(
                Button::new(&mut self.send_button, Text::new("Send"))
                    .padding(10)
                    .on_press(Message::SendMessage),
            );

        // 主界面布局
        let main_content = Column::new()
            .width(Length::Fill)
            .height(Length::Fill)
            .push(message_list)
            .push(input_area);

        Container::new(
            Row::new()
                .spacing(20)
                .push(sidebar)
                .push(main_content)
        )
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

// 模式切换按钮样式
fn mode_button<'a>(
    state: &'a mut button::State,
    label: &str,
    mode: Mode,
    current_mode: &Mode,
) -> Button<'a, Message> {
    let is_active = mode == *current_mode;
    let button_style = if is_active {
        ButtonStyle::Active
    } else {
        ButtonStyle::Inactive
    };

    Button::new(state, Text::new(label))
        .padding(10)
        .style(button_style)
        .on_press(Message::SwitchMode(mode))
}

// 自定义按钮样式
enum ButtonStyle {
    Active,
    Inactive,
}

impl button::StyleSheet for ButtonStyle {
    fn active(&self) -> button::Style {
        match self {
            Self::Active => button::Style {
                background: Some(Color::from_rgb(0.2, 0.5, 1.0).into()),
                border_radius: 5.0,
                text_color: Color::WHITE,
                ..Default::default()
            },
            Self::Inactive => button::Style {
                background: Some(Color::from_rgb(0.8, 0.8, 0.8).into()),
                border_radius: 5.0,
                text_color: Color::BLACK,
                ..Default::default()
            },
        }
    }
}

// 消息气泡样式
struct MessageBubbleStyle;

impl container::StyleSheet for MessageBubbleStyle {
    fn style(&self) -> container::Style {
        container::Style {
            background: Some(Color::from_rgb(0.95, 0.95, 0.95).into()),
            border_radius: 10.0,
            border_width: 1.0,
            border_color: Color::from_rgb(0.8, 0.8, 0.8),
            ..Default::default()
        }
    }
}


fn main() -> iced::Result {
    //初始化kvcache存储输入token,无限loop保持，检测到exit退出对应模式，进行内存回收
    //初始化参数设置
    //前端运行代码

    ChatApp::run(Settings::default())
}
