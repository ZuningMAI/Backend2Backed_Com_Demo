import { createApp } from "vue";
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";
import App from "./App.vue";
import "./style.css";
import * as echarts from "echarts";

(window as any).echarts = echarts;

const app = createApp(App);
app.use(ElementPlus);
app.mount("#app");
