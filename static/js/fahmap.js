$(function() {
    // initialize tooltips
    $.fn.qtip.defaults.style.classes = 'ui-tooltip-bootstrap';
    $.fn.qtip.defaults.style.def = false;
    $.getJSON('./static/json/<%= json %>', function(cities) {
        function map(cont, clustering) {
            var map = kartograph.map(cont);
            map.loadMap('./static/svg/world.svg', function() {
                map.addLayer('mylayer', {});
                map.addLayer('ocean', {});
                var scale = kartograph.scale.sqrt(cities.concat([{ nb_visits: 0 }]), 'nb_visits').range([2, 20]);
                map.addSymbols({
                    type: kartograph.Bubble,
                    data: cities,
                    clustering: clustering,
                    clusteringOpts: {
                        tolerance: 0.01,
                        maxRatio: 0.75
                    },
                    aggregate: function(cities) {
                        var nc = { nb_visits: 0, city_names: [] };
                        $.each(cities, function(i, c) {
                            nc.nb_visits += c.nb_visits;
                            nc.city_names = nc.city_names.concat(c.city_names ? c.city_names : [c.city_name]);
                        });
                        nc.city_name = nc.city_names[0] + ' and ' + (nc.city_names.length-1) + ' others';
                        return nc;
                    },
                    location: function(city) {
                        return [city.long, city.lat];
                    },
                    radius: function(city) {
                        return scale(city.nb_visits);
                    },
                    tooltip: function(city) {
                        msg = '<p>'+city.city_name+'</p>'+city.nb_visits+' donor';
                        if (city.nb_visits > 1) {
                          return msg + 's';
                        }
                        return msg;
                    },
                    sortBy: 'radius desc',
                    style: 'fill:#800; stroke: #fff; fill-opacity: 0.5;',
                });
            }, { padding: -60 });
        }
        map('#map', 'noverlap');
    });
});
