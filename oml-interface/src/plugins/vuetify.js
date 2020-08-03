import Vue from 'vue';
import Vuetify from 'vuetify/lib';

import colors from 'vuetify/lib/util/colors'

Vue.use(Vuetify);

export default new Vuetify({
	theme: {
		themes: {
			light: {
				primary: colors.blueGrey,
				secondary: colors.grey.darken1,
				accent: colors.shades.black,
				error: colors.red.accent3,
				background: colors.indigo.lighten5, // Not automatically applied
			},
			dark: {
				primary: colors.blue.lighten3, 
				background: colors.indigo.base, // If not using lighten/darken, use base to return hex
			},
		},
	},
})
