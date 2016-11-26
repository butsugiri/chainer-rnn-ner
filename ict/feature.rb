#!/usr/bin/env ruby

def output_data(seq, fo, templates)
    seq.each{|h|
        # Extract more characteristics of the input sequence
        head = h['w'][0].clone
        h['iu'] = (h['w'] and head.downcase! != nil).to_s
    }

    for t in 0...seq.length do
        fo.print(seq[t]['y'])
        templates.each{ |template|
            atr = apply_template(seq, t, template)
            if atr != nil
                fo.print("\t"+escape(atr))
            end
        }
        fo.print("\n")
    end
    fo.print("\n")
end

def apply_template(seq, c, template)
    # fo[0]=field, fo[1]=offset
    name = template.map{|fo| 
        if fo != nil
            "#{fo[0]}[#{fo[1]}]"
        end
    }.compact.join("|")

    values = template.map{|fo|
        if fo != nil
            pt = c + fo[1]
            if pt.between?(0,seq.length-1)
                seq[pt][fo[0]]
            else
                false
            end
        end
    }

    if values.include?(false) 
        return nil
    else
        return name + "=" + values.compact.join("|")
    end

end

def escape(src)
    return src.gsub(/:/, "__COLON__")
end

if __FILE__ == $0
    fi = STDIN
    fo = STDOUT

    labels = ['w', 'pos', 'chk', 'iu']

    templates = []
    for l in labels do
        (-2...3).each{|i| templates.push([[l,i],nil]) }
        (-2...2).each{|i| templates.push([[l,i],[l,i+1]]) }
    end

    fnames = ['y', 'w', 'pos', 'chk']
    seq = []
    STDIN.each_line do |line|
        l = line.strip
        if l == ""
            output_data(seq, fo, templates)
            seq = []
        else
            fields = l.split("\t")
            if fields.length != fnames.length
                raise "format error: " + line
            else
                h = {}
                fnames.zip(fields).each{|p|
                    h[p[0]] = p[1]
                }
                seq.push(h)
            end
        end
    end
end
