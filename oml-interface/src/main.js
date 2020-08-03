import Vue from 'vue'
import App from './App.vue'
import router from './router'
import vuetify from './plugins/vuetify';
import VueGraph from 'vue-graph'
import axios from 'axios'
import VueAxios from 'vue-axios'
import GraphLine3D from 'vue-graph/src/components/line3d.js'
import NoteWidget from 'vue-graph/src/widgets/note.js'
import LegendWidget from 'vue-graph/src/widgets/legends.js'
 
Vue.use(VueAxios, axios)
Vue.use(VueGraph)

Vue.component(GraphLine3D.name, GraphLine3D);
Vue.component(NoteWidget.name, NoteWidget);
Vue.component(LegendWidget.name, LegendWidget);

Vue.config.productionTip = false

axios.defaults.baseURL = 'http://ec2-35-176-189-153.eu-west-2.compute.amazonaws.com:4000'

new Vue({
  router,
  vuetify,
  render: h => h(App)
}).$mount('#app')
